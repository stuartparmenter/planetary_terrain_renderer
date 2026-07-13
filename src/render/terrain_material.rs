use crate::{
    debug::DebugTerrain,
    render::{
        DrawTerrainCommand, GpuTerrainView, SetTerrainBindGroup, SetTerrainViewBindGroup,
        TERRAIN_DEPTH_FORMAT, TerrainItem, TerrainTilingPrepassPipelines,
        deferred_gbuffer_targets,
    },
    shaders::{DEFAULT_FRAGMENT_SHADER, DEFAULT_VERTEX_SHADER},
    spawn::{TerrainsToSpawn, spawn_terrains},
    terrain::TerrainComponents,
    terrain_data::{GpuTileAtlas, TileAtlas},
    terrain_shadow::TerrainShadowSettings,
    terrain_view::TerrainViewComponents,
};
use bevy::{
    core_pipeline::prepass::DeferredPrepass,
    light::EnvironmentMapLight,
    material::OpaqueRendererMethod,
    pbr::{
        DefaultOpaqueRendererMethod, ExtractedAtmosphere, MaterialExtractionSystems, MeshPipeline,
        MeshPipelineKey, MeshPipelineSystems, MeshPipelineViewLayoutKey, MeshPipelineViewLayouts,
        RenderMaterialInstance, RenderMaterialInstances, RenderViewLightProbes,
        SetMaterialBindGroup, SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup, ViewKeyCache,
    },
    prelude::*,
    reflect::tuple_struct::TupleStruct,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItemExtraIndex, SetItemPipeline,
            ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::RenderDevice,
        sync_world::MainEntity,
        view::{ExtractedView, RetainedViewEntity},
    },
    shader::{ShaderDefVal, ShaderRef},
};
use std::{hash::Hash, marker::PhantomData};

/// Bevy's mesh view binding-array layout occupies bind group 1 (environment maps, etc.).
pub(crate) const TERRAIN_MATERIAL_BIND_GROUP_INDEX: usize = 4;

#[derive(PartialEq, Eq, Clone, Hash)]
pub struct TerrainPipelineKey {
    pub flags: TerrainPipelineFlags,
    pub color_target_format: TextureFormat,
    /// The mesh-view bind group layout key Bevy computed for this view
    /// (from [`ViewKeyCache`]). Reused verbatim so the terrain's group-0
    /// layout always matches the view's `mesh_view_bind_group`.
    pub view_layout_key: MeshPipelineViewLayoutKey,
    /// The view's `SHADOW_FILTER_METHOD_*` bits from Bevy's
    /// [`MeshPipelineKey`], forwarded so `sample_shadow_map` specializes to
    /// the same filter as the mesh pipeline — with no method defined its
    /// fallback returns 0.0 (fully shadowed).
    pub shadow_filter_method: MeshPipelineKey,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct TerrainPipelineFlags: u32 {
        const NONE               = 0;
        const SPHERICAL          = 1 <<  0;
        const WIREFRAME          = 1 <<  1;
        const SHOW_DATA_LOD      = 1 <<  2;
        const SHOW_GEOMETRY_LOD  = 1 <<  3;
        const SHOW_TILE_TREE     = 1 <<  4;
        const SHOW_PIXELS        = 1 <<  5;
        const SHOW_UV            = 1 <<  6;
        const SHOW_NORMALS       = 1 <<  7;
        const MORPH              = 1 <<  8;
        const BLEND              = 1 <<  9;
        const TILE_TREE_LOD      = 1 << 10;
        const LIGHTING           = 1 << 11;
        const SAMPLE_GRAD        = 1 << 12;
        const HIGH_PRECISION     = 1 << 13;
        const TEST1              = 1 << 14;
        const TEST2              = 1 << 15;
        const TEST3              = 1 << 16;
        const HDR                = 1 << 17;
        const ATMOSPHERE         = 1 << 18;
        const ENVIRONMENT_MAP    = 1 << 19;
        const TERRAIN_SHADOW     = 1 << 20;
        const DEFERRED           = 1 << 21;
        const MSAA_RESERVED_BITS = TerrainPipelineFlags::MSAA_MASK_BITS << TerrainPipelineFlags::MSAA_SHIFT_BITS;
    }
}

impl TerrainPipelineFlags {
    const MSAA_MASK_BITS: u32 = 0b111111;
    const MSAA_SHIFT_BITS: u32 = 32 - 6;

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits = ((msaa_samples - 1) & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        TerrainPipelineFlags::from_bits(msaa_bits).unwrap()
    }

    pub fn from_debug(debug: &DebugTerrain) -> Self {
        let mut key = TerrainPipelineFlags::NONE;

        if debug.wireframe {
            key |= TerrainPipelineFlags::WIREFRAME;
        }
        if debug.show_data_lod {
            key |= TerrainPipelineFlags::SHOW_DATA_LOD;
        }
        if debug.show_geometry_lod {
            key |= TerrainPipelineFlags::SHOW_GEOMETRY_LOD;
        }
        if debug.show_tile_tree {
            key |= TerrainPipelineFlags::SHOW_TILE_TREE;
        }
        if debug.show_pixels {
            key |= TerrainPipelineFlags::SHOW_PIXELS;
        }
        if debug.show_uv {
            key |= TerrainPipelineFlags::SHOW_UV;
        }
        if debug.show_normals {
            key |= TerrainPipelineFlags::SHOW_NORMALS;
        }
        if debug.morph {
            key |= TerrainPipelineFlags::MORPH;
        }
        if debug.blend {
            key |= TerrainPipelineFlags::BLEND;
        }
        if debug.tile_tree_lod {
            key |= TerrainPipelineFlags::TILE_TREE_LOD;
        }
        if debug.lighting {
            key |= TerrainPipelineFlags::LIGHTING;
        }
        if debug.sample_grad {
            key |= TerrainPipelineFlags::SAMPLE_GRAD;
        }
        if debug.high_precision {
            key |= TerrainPipelineFlags::HIGH_PRECISION;
        }
        if debug.test1 {
            key |= TerrainPipelineFlags::TEST1;
        }
        if debug.test2 {
            key |= TerrainPipelineFlags::TEST2;
        }
        if debug.test3 {
            key |= TerrainPipelineFlags::TEST3;
        }

        key
    }

    pub fn msaa_samples(&self) -> u32 {
        ((self.bits() >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS) + 1
    }

    pub fn polygon_mode(&self) -> PolygonMode {
        match self.contains(TerrainPipelineFlags::WIREFRAME) {
            true => PolygonMode::Line,
            false => PolygonMode::Fill,
        }
    }

    pub fn shader_defs(&self) -> Vec<ShaderDefVal> {
        let mut shader_defs = Vec::new();

        if self.contains(TerrainPipelineFlags::SPHERICAL) {
            shader_defs.push("SPHERICAL".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_DATA_LOD) {
            shader_defs.push("SHOW_DATA_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_GEOMETRY_LOD) {
            shader_defs.push("SHOW_GEOMETRY_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_TILE_TREE) {
            shader_defs.push("SHOW_TILE_TREE".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_PIXELS) {
            shader_defs.push("SHOW_PIXELS".into())
        }
        if self.contains(TerrainPipelineFlags::SHOW_UV) {
            shader_defs.push("SHOW_UV".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_NORMALS) {
            shader_defs.push("SHOW_NORMALS".into())
        }
        if self.contains(TerrainPipelineFlags::MORPH) {
            shader_defs.push("MORPH".into());
        }
        if self.contains(TerrainPipelineFlags::BLEND) {
            shader_defs.push("BLEND".into());
        }
        if self.contains(TerrainPipelineFlags::TILE_TREE_LOD) {
            shader_defs.push("TILE_TREE_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::LIGHTING) {
            shader_defs.push("LIGHTING".into());
        }
        if self.contains(TerrainPipelineFlags::SAMPLE_GRAD) {
            shader_defs.push("SAMPLE_GRAD".into());
        }

        if self.contains(TerrainPipelineFlags::HIGH_PRECISION) {
            shader_defs.push("HIGH_PRECISION".into());
        }
        if self.contains(TerrainPipelineFlags::TEST1) {
            shader_defs.push("TEST1".into());
        }
        if self.contains(TerrainPipelineFlags::TEST2) {
            shader_defs.push("TEST2".into());
        }
        if self.contains(TerrainPipelineFlags::TEST3) {
            shader_defs.push("TEST3".into());
        }
        if self.contains(TerrainPipelineFlags::HDR) {
            shader_defs.push("HDR".into());
        }
        if self.contains(TerrainPipelineFlags::ATMOSPHERE) {
            shader_defs.push("ATMOSPHERE".into());
        }
        if self.contains(TerrainPipelineFlags::ENVIRONMENT_MAP) {
            shader_defs.push("ENVIRONMENT_MAP".into());
        }
        if self.contains(TerrainPipelineFlags::TERRAIN_SHADOW) {
            shader_defs.push("TERRAIN_SHADOW".into());
        }
        if self.contains(TerrainPipelineFlags::DEFERRED) {
            shader_defs.push("DEFERRED_PREPASS".into());
        }

        shader_defs
    }
}

/// Whether the terrain renders through the deferred G-buffer on a view:
/// the view has a deferred prepass AND `OpaqueRendererMethod::Auto`
/// currently resolves to deferred — the same rule Bevy's opaque materials
/// follow (`SolariPlugins` sets the default to deferred globally). The
/// single definition is load-bearing: `queue_terrain`, both render passes,
/// the motion pass, and the target allocation must agree bit-for-bit or
/// they skew against each other. The resource's inner method has no public
/// accessor, so it is read through its `Reflect` tuple-struct impl.
pub(crate) fn terrain_renders_deferred(
    has_deferred_prepass: bool,
    method: Option<&DefaultOpaqueRendererMethod>,
) -> bool {
    has_deferred_prepass
        && method
            .and_then(|method| method.field(0))
            .and_then(|field| field.try_downcast_ref::<OpaqueRendererMethod>())
            .is_some_and(|method| *method == OpaqueRendererMethod::Deferred)
}

fn extract_terrain_materials<M: Material>(
    mut material_instances: ResMut<RenderMaterialInstances>,
    terrains: Extract<Query<(Entity, &MeshMaterial3d<M>), With<TileAtlas>>>,
) {
    let last_change_tick = material_instances.current_change_tick;

    for (entity, material) in &terrains {
        material_instances.instances.insert(
            entity.into(),
            RenderMaterialInstance {
                asset_id: material.id().untyped(),
                last_change_tick,
            },
        );
    }
}

/// The pipeline used to render the terrain entities.
#[derive(Resource)]
pub struct TerrainRenderPipeline<M: Material> {
    mesh_view_layouts: MeshPipelineViewLayouts,
    binding_arrays_are_usable: bool,
    terrain_layout: BindGroupLayoutDescriptor,
    terrain_view_layout: BindGroupLayoutDescriptor,
    terrain_view_layout_debug: BindGroupLayoutDescriptor,
    material_layout: BindGroupLayoutDescriptor,
    vertex_shader: Handle<Shader>,
    fragment_shader: Handle<Shader>,
    marker: PhantomData<M>,
}

impl<M: Material> FromWorld for TerrainRenderPipeline<M> {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let mesh_pipeline = world.resource::<MeshPipeline>();
        let prepass_pipelines = world.resource::<TerrainTilingPrepassPipelines>();

        let vertex_shader = match M::vertex_shader() {
            ShaderRef::Default => world.load_asset(DEFAULT_VERTEX_SHADER),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => world.load_asset(path),
        };

        let fragment_shader = match M::fragment_shader() {
            ShaderRef::Default => world.load_asset(DEFAULT_FRAGMENT_SHADER),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => world.load_asset(path),
        };

        Self {
            mesh_view_layouts: mesh_pipeline.view_layouts.clone(),
            binding_arrays_are_usable: mesh_pipeline.binding_arrays_are_usable,
            terrain_layout: prepass_pipelines.terrain_layout.clone(),
            terrain_view_layout: prepass_pipelines.terrain_view_layout.clone(),
            terrain_view_layout_debug: prepass_pipelines.terrain_view_layout_debug.clone(),
            material_layout: M::bind_group_layout_descriptor(device),
            vertex_shader,
            fragment_shader,
            marker: PhantomData,
        }
    }
}

impl<M: Material> SpecializedRenderPipeline for TerrainRenderPipeline<M> {
    type Key = TerrainPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = key.flags.shader_defs();

        if key.flags.msaa_samples() > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }

        let view_layout_key = key.view_layout_key;
        if self.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHT_PROBES_IN_ARRAY".into());
        }

        if key.shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2 {
            shader_defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
        } else if key.shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN {
            shader_defs.push("SHADOW_FILTER_METHOD_GAUSSIAN".into());
        } else if key.shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL {
            shader_defs.push("SHADOW_FILTER_METHOD_TEMPORAL".into());
        }

        let view_layout = self.mesh_view_layouts.get_view_layout(view_layout_key);
        let mut bind_group_layouts = vec![
            view_layout.main_layout.clone(),
            view_layout.binding_array_layout.clone(),
            self.terrain_layout.clone(),
        ];
        bind_group_layouts.push(
            if key.flags.contains(TerrainPipelineFlags::SHOW_TILE_TREE)
                && !key.flags.contains(TerrainPipelineFlags::ATMOSPHERE)
            {
                self.terrain_view_layout_debug.clone()
            } else {
                self.terrain_view_layout.clone()
            },
        );
        bind_group_layouts.push(self.material_layout.clone());

        let mut vertex_shader_defs = shader_defs.clone();
        vertex_shader_defs.push("VERTEX".into());
        let mut fragment_shader_defs = shader_defs.clone();
        fragment_shader_defs.push("FRAGMENT".into());

        // On deferred views the fragment writes Bevy's G-buffer instead of
        // the view target: `terrain_deferred_pass` renders target slots 0
        // and 1 into the private `TerrainDeferredTargets`. Everything else,
        // notably the group-0 layout derived from `view_layout_key` and the
        // private depth/stencil block, is shared with the forward pipeline.
        let targets = if key.flags.contains(TerrainPipelineFlags::DEFERRED) {
            deferred_gbuffer_targets().to_vec()
        } else {
            vec![Some(ColorTargetState {
                format: key.color_target_format,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })]
        };

        RenderPipelineDescriptor {
            label: None,
            layout: bind_group_layouts,
            immediate_size: 0,
            vertex: VertexState {
                shader: self.vertex_shader.clone(),
                entry_point: Some("vertex".into()),
                shader_defs: vertex_shader_defs,
                buffers: Vec::new(),
                constants: vec![],
            },
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: key.flags.polygon_mode(),
                conservative: false,
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(IndexFormat::Uint32),
            },
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs: fragment_shader_defs,
                entry_point: Some("fragment".into()),
                targets,
                constants: vec![],
            }),
            depth_stencil: Some(DepthStencilState {
                format: TERRAIN_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Greater),
                stencil: StencilState {
                    front: StencilFaceState {
                        compare: CompareFunction::GreaterEqual,
                        fail_op: StencilOperation::Keep,
                        depth_fail_op: StencilOperation::Keep,
                        pass_op: StencilOperation::Replace,
                    },
                    back: StencilFaceState::IGNORE,
                    read_mask: !0,
                    write_mask: !0,
                },
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.flags.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: false,
        }
    }
}

pub fn init_terrain_render_pipeline<M: Material>(world: &mut World) {
    world.init_resource::<TerrainRenderPipeline<M>>();
}

/// The draw function of the terrain. It sets the pipeline and the bind groups and then issues the
/// draw call.
pub(crate) type DrawTerrain = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetTerrainBindGroup<2>,
    SetTerrainViewBindGroup<3>,
    SetMaterialBindGroup<TERRAIN_MATERIAL_BIND_GROUP_INDEX>,
    DrawTerrainCommand,
);

/// Queses all terrain entities for rendering via the terrain pipeline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_terrain<M: Material>(
    draw_functions: Res<DrawFunctions<TerrainItem>>,
    debug: Option<Res<DebugTerrain>>,
    shadow_settings: Res<TerrainShadowSettings>,
    pipeline_cache: Res<PipelineCache>,
    terrain_pipeline: Res<TerrainRenderPipeline<M>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TerrainRenderPipeline<M>>>,
    mut terrain_phases: ResMut<ViewSortedRenderPhases<TerrainItem>>,
    gpu_tile_atlases: Res<TerrainComponents<GpuTileAtlas>>,
    gpu_terrain_views: Res<TerrainViewComponents<GpuTerrainView>>,
    view_key_cache: Res<ViewKeyCache>,
    default_opaque_renderer_method: Option<Res<DefaultOpaqueRendererMethod>>,
    mut views: Query<(
        MainEntity,
        &Msaa,
        &ExtractedView,
        Has<ExtractedAtmosphere>,
        Has<RenderViewLightProbes<EnvironmentMapLight>>,
        Has<DeferredPrepass>,
    )>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let draw_function = draw_functions.read().get_id::<DrawTerrain>().unwrap();

    for (view, msaa, extracted_view, has_atmosphere, has_environment_maps, has_deferred_prepass) in
        &mut views
    {
        let Some(terrain_phase) = terrain_phases.get_mut(&RetainedViewEntity {
            main_entity: view.into(),
            auxiliary_entity: Entity::PLACEHOLDER.into(),
            subview_index: 0,
        }) else {
            continue;
        };

        for (&terrain, gpu_tile_atlas) in gpu_tile_atlases.iter() {
            let Some(gpu_terrain_view) = gpu_terrain_views.get(&(terrain, view)) else {
                continue;
            };

            // Reuse the exact mesh-view layout Bevy computed for this view
            // (prepass slots, atmosphere, env map, STBN, …) so the terrain's
            // group-0 layout always matches the `mesh_view_bind_group` —
            // rather than re-deriving it flag by flag, which drifts as views
            // gain features.
            let Some(view_key) = view_key_cache.get(&extracted_view.retained_view_entity) else {
                continue;
            };
            let view_layout_key = MeshPipelineViewLayoutKey::from(*view_key);

            let mut flags = TerrainPipelineFlags::from_msaa_samples(msaa.samples());
            if gpu_tile_atlas.is_spherical {
                flags |= TerrainPipelineFlags::SPHERICAL;
            }

            let hdr = extracted_view.target_format != TextureFormat::Rgba8UnormSrgb
                && extracted_view.target_format != TextureFormat::Rgba8Unorm;
            if hdr {
                flags |= TerrainPipelineFlags::HDR;
            }
            if has_atmosphere {
                flags |= TerrainPipelineFlags::ATMOSPHERE;
            }
            if has_environment_maps {
                flags |= TerrainPipelineFlags::ENVIRONMENT_MAP;
            }
            if shadow_settings.enabled {
                flags |= TerrainPipelineFlags::TERRAIN_SHADOW;
            }

            if let Some(debug) = &debug {
                flags |= TerrainPipelineFlags::from_debug(debug);
            } else {
                flags |= TerrainPipelineFlags::LIGHTING
                    | TerrainPipelineFlags::MORPH
                    | TerrainPipelineFlags::BLEND
                    | TerrainPipelineFlags::SAMPLE_GRAD
                    | TerrainPipelineFlags::HIGH_PRECISION;
            }

            // On deferred views the same phase holds deferred-pipeline
            // items, which `terrain_deferred_pass` renders and
            // `terrain_pass` skips. DEFERRED implies LIGHTING — applied
            // after the debug flags so a lighting-off toggle can't strip
            // it: the G-buffer `FragmentOutput` has no color slot for the
            // unlit path, and the packed PbrInput comes from the shared
            // LIGHTING assembly.
            if terrain_renders_deferred(
                has_deferred_prepass,
                default_opaque_renderer_method.as_deref(),
            ) {
                flags |= TerrainPipelineFlags::DEFERRED | TerrainPipelineFlags::LIGHTING;
            }

            let key = TerrainPipelineKey {
                flags,
                color_target_format: extracted_view.target_format,
                view_layout_key,
                shadow_filter_method: view_key
                    .intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS),
            };

            let pipeline = pipelines.specialize(&pipeline_cache, &terrain_pipeline, key);

            terrain_phase.add_transient(TerrainItem {
                representative_entity: (terrain, terrain.into()), // technically wrong
                draw_function,
                pipeline,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                order: gpu_terrain_view.order,
            })
        }
    }
}

/// This plugin adds a custom material for a terrain.
///
/// It can be used to render the terrain using a custom vertex and fragment shader.
pub struct TerrainMaterialPlugin<M: Material>(PhantomData<M>);

impl<M: Material> Default for TerrainMaterialPlugin<M> {
    fn default() -> Self {
        Self(default())
    }
}

impl<M: Material + Clone> Plugin for TerrainMaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<M>::default())
            .insert_resource(TerrainsToSpawn::<M>(vec![]))
            .add_systems(PostUpdate, spawn_terrains::<M>);

        app.sub_app_mut(RenderApp)
            .add_render_command::<TerrainItem, DrawTerrain>()
            .init_resource::<SpecializedRenderPipelines<TerrainRenderPipeline<M>>>()
            .add_systems(
                ExtractSchedule,
                extract_terrain_materials::<M>.in_set(MaterialExtractionSystems),
            )
            .add_systems(
                RenderStartup,
                init_terrain_render_pipeline::<M>.after(MeshPipelineSystems),
            )
            .add_systems(
                Render,
                queue_terrain::<M>.in_set(RenderSystems::QueueMeshes),
            );
    }
}
