use crate::{
    debug::DebugTerrain,
    math::{TileCoordinate, ViewCoordinate},
    render::{TerrainTilingPrepassPipelines, terrain_shadow::GpuTerrainShadow},
    terrain_data::{TileTree, TileTreeEntry},
    terrain_shadow::TerrainShadowUniform,
    terrain_view::TerrainViewComponents,
};
use bevy::{
    ecs::{
        query::ROQueryItem,
        system::{StaticSystemParam, SystemParamItem, lifetimeless::SRes},
    },
    pbr::ExtractedAtmosphere,
    prelude::*,
    render::{
        Extract,
        render_asset::RenderAssets,
        render_phase::{PhaseItem, RenderCommand, RenderCommandResult, TrackedRenderPass},
        render_resource::{binding_types::*, *},
        renderer::RenderDevice,
        storage::{GpuShaderBuffer, ShaderBuffer},
        sync_world::MainEntity,
    },
};

#[derive(AsBindGroup)]
pub struct IndirectBindGroup {
    #[storage(0, visibility(compute), buffer)]
    pub(crate) indirect: Buffer,
}

#[derive(AsBindGroup)]
pub struct PrepassViewBindGroup {
    #[storage(0, visibility(compute), read_only)]
    pub(crate) terrain_view: Handle<ShaderBuffer>,
    #[storage(1, visibility(compute))]
    pub(crate) approximate_height: Handle<ShaderBuffer>,
    #[storage(2, visibility(compute), read_only)]
    pub(crate) tile_tree: Handle<ShaderBuffer>,
    #[storage(3, visibility(compute), buffer)]
    pub(crate) final_tiles: Buffer,
    #[storage(4, visibility(compute), buffer)]
    pub(crate) temporary_tiles: Buffer,
    #[storage(5, visibility(compute), buffer)]
    pub(crate) state: Buffer,
}

#[derive(AsBindGroup)]
pub struct TerrainViewBindGroup {
    // Todo: replace with updatable uniform buffer
    #[storage(0, visibility(vertex, fragment), read_only)]
    pub(crate) terrain_view: Handle<ShaderBuffer>,
    #[storage(1, visibility(vertex), read_only)]
    pub(crate) approximate_height: Handle<ShaderBuffer>,
    #[storage(2, visibility(vertex, fragment), read_only)]
    pub(crate) tile_tree: Handle<ShaderBuffer>,
    #[storage(3, visibility(vertex, fragment), read_only, buffer)]
    pub(crate) geometry_tiles: Buffer,
}

/// Same bindings as [`TerrainViewBindGroup`], but exposes `tile_tree` to the fragment stage for
/// debug visualizations (`SHOW_TILE_TREE`).
#[derive(AsBindGroup)]
pub struct TerrainViewBindGroupDebug {
    #[storage(0, visibility(vertex, fragment), read_only)]
    pub(crate) terrain_view: Handle<ShaderBuffer>,
    #[storage(1, visibility(vertex), read_only)]
    pub(crate) approximate_height: Handle<ShaderBuffer>,
    #[storage(2, visibility(vertex, fragment), read_only)]
    pub(crate) tile_tree: Handle<ShaderBuffer>,
    #[storage(3, visibility(vertex, fragment), read_only, buffer)]
    pub(crate) geometry_tiles: Buffer,
}

/// Appends the terrain shadow map bindings (texture, sampler, params) to the
/// derived terrain view layout, so the fragment shader can sample the shadow map
/// under the `TERRAIN_SHADOW` shader def.
pub(crate) fn extend_terrain_view_layout(
    mut descriptor: BindGroupLayoutDescriptor,
) -> BindGroupLayoutDescriptor {
    descriptor.entries.extend([
        texture_2d(TextureSampleType::Float { filterable: true }).build(4, ShaderStages::FRAGMENT),
        sampler(SamplerBindingType::Filtering).build(5, ShaderStages::FRAGMENT),
        uniform_buffer::<TerrainShadowUniform>(false).build(6, ShaderStages::FRAGMENT),
    ]);
    descriptor
}

#[derive(ShaderType)]
pub(crate) struct GeometryTile {
    face: u32,
    lod: u32,
    xy: UVec2,
    view_distances: Vec4,
    morph_ratios: Vec4,
}

#[derive(ShaderType)]
pub(crate) struct Indirect {
    x_or_vertex_count: u32,
    y_or_instance_count: u32,
    z_or_base_vertex: u32,
    base_instance: u32,
}

#[derive(ShaderType)]
pub(crate) struct PrepassState {
    tile_count: u32,
    counter: i32,
    child_index: i32,
    final_index: i32,
}

#[derive(Default, ShaderType)]
pub struct TileTreeUniform {
    #[shader(size(runtime))]
    pub(crate) entries: Vec<TileTreeEntry>,
}

#[derive(ShaderType)]
pub(crate) struct TerrainViewUniform {
    tree_size: u32,
    geometry_tile_count: u32,
    grid_size: f32,
    vertices_per_row: u32,
    vertices_per_tile: u32,
    morph_distance: f32,
    blend_distance: f32,
    load_distance: f32,
    subdivision_distance: f32,
    morph_range: f32,
    blend_range: f32,
    precision_distance: f32,
    face: u32,
    lod: u32,
    coordinates: [ViewCoordinate; 6],
    world_position: Vec3,
    half_spaces: [Vec4; 6],
    surface_approximation: [crate::math::SurfaceApproximation; 6],
}

impl From<&TileTree> for TerrainViewUniform {
    fn from(tile_tree: &TileTree) -> Self {
        TerrainViewUniform {
            tree_size: tile_tree.tree_size,
            geometry_tile_count: tile_tree.geometry_tile_count,
            grid_size: tile_tree.grid_size as f32,
            vertices_per_row: 2 * (tile_tree.grid_size + 2),
            vertices_per_tile: 2 * tile_tree.grid_size * (tile_tree.grid_size + 2),
            morph_distance: tile_tree.morph_distance as f32,
            blend_distance: tile_tree.blend_distance as f32,
            load_distance: tile_tree.load_distance as f32,
            subdivision_distance: tile_tree.subdivision_distance as f32,
            precision_distance: tile_tree.precision_distance as f32,
            morph_range: tile_tree.morph_range,
            blend_range: tile_tree.blend_range,
            face: tile_tree.view_face,
            lod: tile_tree.view_lod,
            coordinates: tile_tree
                .view_coordinates
                .map(|view_coordinate| ViewCoordinate::new(view_coordinate, tile_tree.view_lod)),
            world_position: tile_tree.view_world_position,
            half_spaces: tile_tree.half_spaces,

            surface_approximation: tile_tree.surface_approximation.clone(),
        }
    }
}

pub struct GpuTerrainView {
    pub(crate) order: u32,
    pub(crate) refinement_count: u32,
    pub(crate) indirect_buffer: Buffer,
    pub(crate) indirect_bind_group: Option<BindGroup>,
    pub(crate) prepass_view_bind_group: Option<BindGroup>,
    pub(crate) terrain_view_bind_group: Option<BindGroup>,
    terrain_view_debug_layout: Option<bool>,
    /// Ids of the render-asset buffers `prepass_view_bind_group` was built
    /// from — those can be reallocated on re-prepare, the directly-owned
    /// buffers never change.
    prepass_view_buffers: Option<(BufferId, BufferId, BufferId)>,

    indirect: IndirectBindGroup,
    prepass_view: PrepassViewBindGroup,
    terrain_view: TerrainViewBindGroup,
}

impl GpuTerrainView {
    fn new(device: &RenderDevice, tile_tree: &TileTree) -> Self {
        // Todo: figure out a better way of limiting the tile buffer size

        let tiles = device.create_buffer(&BufferDescriptor {
            label: None,
            size: GeometryTile::min_size().get() * tile_tree.geometry_tile_count as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let temporary_tiles = device.create_buffer(&BufferDescriptor {
            label: None,
            size: TileCoordinate::min_size().get() * tile_tree.geometry_tile_count as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let state = device.create_buffer(&BufferDescriptor {
            label: None,
            size: PrepassState::min_size().get(),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let indirect = device.create_buffer(&BufferDescriptor {
            label: None,
            size: Indirect::min_size().get(),
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let prepare_prepass = IndirectBindGroup {
            indirect: indirect.clone(),
        };
        let refine_tiles = PrepassViewBindGroup {
            terrain_view: tile_tree.terrain_view_buffer.clone(),
            approximate_height: tile_tree.approximate_height_buffer.clone(),
            tile_tree: tile_tree.tile_tree_buffer.clone(),
            final_tiles: tiles.clone(),
            temporary_tiles,
            state,
        };
        let terrain_view = TerrainViewBindGroup {
            terrain_view: tile_tree.terrain_view_buffer.clone(),
            approximate_height: tile_tree.approximate_height_buffer.clone(),
            tile_tree: tile_tree.tile_tree_buffer.clone(),
            geometry_tiles: tiles,
        };

        Self {
            order: tile_tree.order,
            refinement_count: tile_tree.refinement_count,
            indirect_buffer: indirect,
            indirect: prepare_prepass,
            prepass_view: refine_tiles,
            terrain_view,
            indirect_bind_group: None,
            prepass_view_bind_group: None,
            terrain_view_bind_group: None,
            terrain_view_debug_layout: None,
            prepass_view_buffers: None,
        }
    }

    pub(crate) fn initialize(
        device: Res<RenderDevice>,
        mut gpu_terrain_views: ResMut<TerrainViewComponents<GpuTerrainView>>,
        tile_trees: Extract<Res<TerrainViewComponents<TileTree>>>,
    ) {
        for (&(terrain, view), tile_tree) in tile_trees.iter() {
            if gpu_terrain_views.contains_key(&(terrain, view)) {
                continue;
            }

            gpu_terrain_views.insert((terrain, view), GpuTerrainView::new(&device, tile_tree));
        }
    }

    pub(crate) fn prepare_terrain_view(
        device: Res<RenderDevice>,
        pipeline_cache: Res<PipelineCache>,
        prepass_pipeline: Res<TerrainTilingPrepassPipelines>,
        debug: Option<Res<DebugTerrain>>,
        atmosphere_cameras: Query<Has<ExtractedAtmosphere>, With<Camera3d>>,
        buffers: Res<RenderAssets<GpuShaderBuffer>>,
        gpu_terrain_shadows: Res<TerrainViewComponents<GpuTerrainShadow>>,
        mut gpu_terrain_views: ResMut<TerrainViewComponents<GpuTerrainView>>,
    ) {
        let show_tile_tree = debug.is_some_and(|debug| debug.show_tile_tree);
        let atmosphere_active = atmosphere_cameras.iter().any(|has| has);
        let use_debug_layout = show_tile_tree && !atmosphere_active;

        for (&(terrain, view), gpu_terrain_view) in gpu_terrain_views.iter_mut() {
            let terrain_view = &gpu_terrain_view.terrain_view;

            let (
                Some(terrain_view_buffer),
                Some(approximate_height_buffer),
                Some(tile_tree_buffer),
                Some(gpu_terrain_shadow),
            ) = (
                buffers.get(&terrain_view.terrain_view),
                buffers.get(&terrain_view.approximate_height),
                buffers.get(&terrain_view.tile_tree),
                gpu_terrain_shadows.get(&(terrain, view)),
            )
            else {
                gpu_terrain_view.terrain_view_bind_group = None;
                gpu_terrain_view.terrain_view_debug_layout = None;
                continue;
            };

            if gpu_terrain_view.terrain_view_bind_group.is_some()
                && gpu_terrain_view.terrain_view_debug_layout == Some(use_debug_layout)
            {
                continue;
            }

            let layout = if use_debug_layout {
                &prepass_pipeline.terrain_view_layout_debug
            } else {
                &prepass_pipeline.terrain_view_layout
            };

            gpu_terrain_view.terrain_view_bind_group = Some(device.create_bind_group(
                "terrain_view_bind_group",
                &pipeline_cache.get_bind_group_layout(layout),
                &BindGroupEntries::sequential((
                    terrain_view_buffer.buffer.as_entire_binding(),
                    approximate_height_buffer.buffer.as_entire_binding(),
                    tile_tree_buffer.buffer.as_entire_binding(),
                    terrain_view.geometry_tiles.as_entire_binding(),
                    &gpu_terrain_shadow.texture_view,
                    &gpu_terrain_shadow.sampler,
                    &gpu_terrain_shadow.params_buffer,
                )),
            ));
            gpu_terrain_view.terrain_view_debug_layout = Some(use_debug_layout);
        }
    }

    pub(crate) fn prepare_indirect(
        device: Res<RenderDevice>,
        pipeline_cache: Res<PipelineCache>,
        prepass_pipeline: Res<TerrainTilingPrepassPipelines>,
        mut gpu_terrain_views: ResMut<TerrainViewComponents<GpuTerrainView>>,
        mut param: StaticSystemParam<<IndirectBindGroup as AsBindGroup>::Param>,
    ) {
        for gpu_terrain_view in &mut gpu_terrain_views.values_mut() {
            let bind_group = &mut gpu_terrain_view.indirect_bind_group;

            if bind_group.is_none() {
                *bind_group = gpu_terrain_view
                    .indirect
                    .as_bind_group(
                        &prepass_pipeline.indirect_layout,
                        &device,
                        &pipeline_cache,
                        &mut param,
                    )
                    .ok()
                    .map(|b| b.bind_group);
            }
        }
    }

    pub(crate) fn prepare_refine_tiles(
        device: Res<RenderDevice>,
        pipeline_cache: Res<PipelineCache>,
        prepass_pipeline: Res<TerrainTilingPrepassPipelines>,
        buffers: Res<RenderAssets<GpuShaderBuffer>>,
        mut gpu_terrain_views: ResMut<TerrainViewComponents<GpuTerrainView>>,
    ) {
        for gpu_terrain_view in gpu_terrain_views.values_mut() {
            let prepass_view = &gpu_terrain_view.prepass_view;

            let (Some(terrain_view_buffer), Some(approximate_height_buffer), Some(tile_tree_buffer)) = (
                buffers.get(&prepass_view.terrain_view),
                buffers.get(&prepass_view.approximate_height),
                buffers.get(&prepass_view.tile_tree),
            ) else {
                gpu_terrain_view.prepass_view_bind_group = None;
                gpu_terrain_view.prepass_view_buffers = None;
                continue;
            };

            let buffer_ids = (
                terrain_view_buffer.buffer.id(),
                approximate_height_buffer.buffer.id(),
                tile_tree_buffer.buffer.id(),
            );
            if gpu_terrain_view.prepass_view_bind_group.is_some()
                && gpu_terrain_view.prepass_view_buffers == Some(buffer_ids)
            {
                continue;
            }

            // Entries mirror the `AsBindGroup` field order of
            // `PrepassViewBindGroup` (bindings 0..=5).
            gpu_terrain_view.prepass_view_bind_group = Some(device.create_bind_group(
                "prepass_view_bind_group",
                &pipeline_cache.get_bind_group_layout(&prepass_pipeline.prepass_view_layout),
                &BindGroupEntries::sequential((
                    terrain_view_buffer.buffer.as_entire_binding(),
                    approximate_height_buffer.buffer.as_entire_binding(),
                    tile_tree_buffer.buffer.as_entire_binding(),
                    prepass_view.final_tiles.as_entire_binding(),
                    prepass_view.temporary_tiles.as_entire_binding(),
                    prepass_view.state.as_entire_binding(),
                )),
            ));
            gpu_terrain_view.prepass_view_buffers = Some(buffer_ids);
        }
    }
}

pub struct SetTerrainViewBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetTerrainViewBindGroup<I> {
    type Param = SRes<TerrainViewComponents<GpuTerrainView>>;
    type ViewQuery = MainEntity;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        gpu_terrain_views: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_terrain_view = &gpu_terrain_views.into_inner()[&(item.main_entity().id(), view)];

        if let Some(bind_group) = &gpu_terrain_view.terrain_view_bind_group {
            pass.set_bind_group(I, bind_group, &[]);
            RenderCommandResult::Success
        } else {
            RenderCommandResult::Skip
        }
    }
}

pub(crate) struct DrawTerrainCommand;

impl<P: PhaseItem> RenderCommand<P> for DrawTerrainCommand {
    type Param = SRes<TerrainViewComponents<GpuTerrainView>>;
    type ViewQuery = MainEntity;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        gpu_terrain_views: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_terrain_view = &gpu_terrain_views.into_inner()[&(item.main_entity().id(), view)];

        pass.set_stencil_reference(gpu_terrain_view.order);
        pass.draw_indirect(&gpu_terrain_view.indirect_buffer, 0);

        RenderCommandResult::Success
    }
}
