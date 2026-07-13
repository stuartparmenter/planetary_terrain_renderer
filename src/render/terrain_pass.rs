use crate::render::terrain_material::terrain_renders_deferred;
use crate::shaders::{DEFERRED_COMPOSITE_SHADER, DEPTH_COPY_SHADER, TERRAIN_MOTION_SHADER};
use crate::terrain::TerrainComponents;
use crate::terrain_data::GpuTileAtlas;
use bevy::{
    camera::{MainPassResolutionOverride, Viewport},
    core_pipeline::{
        FullscreenShader,
        core_3d::CORE_3D_DEPTH_FORMAT,
        deferred::{DEFERRED_LIGHTING_PASS_ID_FORMAT, DEFERRED_PREPASS_FORMAT},
        prepass::{
            DeferredPrepass, MOTION_VECTOR_PREPASS_FORMAT, PreviousViewData,
            PreviousViewUniformOffset, PreviousViewUniforms, ViewPrepassTextures,
        },
    },
    ecs::entity::EntityHash,
    pbr::DefaultOpaqueRendererMethod,
    prelude::*,
    render::{
        Extract,
        camera::ExtractedCamera,
        render_phase::{
            CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem, PhaseItemExtraIndex,
            SortedPhaseItem, ViewSortedRenderPhases,
        },
        render_resource::{
            binding_types::{
                texture_2d, texture_depth_2d, texture_depth_2d_multisampled, uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        sync_world::MainEntity,
        texture::{CachedTexture, TextureCache},
        view::{
            ExtractedView, RetainedViewEntity, ViewDepthTexture, ViewTarget, ViewUniform,
            ViewUniformOffset, ViewUniforms,
        },
    },
};
use indexmap::IndexMap;
use std::ops::Range;

pub(crate) const TERRAIN_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32FloatStencil8;

pub struct TerrainItem {
    pub representative_entity: (Entity, MainEntity),
    pub draw_function: DrawFunctionId,
    pub pipeline: CachedRenderPipelineId,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
    pub order: u32,
}

impl PhaseItem for TerrainItem {
    const AUTOMATIC_BATCHING: bool = false;

    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    #[inline]
    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for TerrainItem {
    type SortKey = u32;

    fn sort_key(&self) -> Self::SortKey {
        u32::MAX - self.order
    }

    fn recalculate_sort_keys(
        _items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _view: &ExtractedView,
    ) {
    }

    fn indexed(&self) -> bool {
        false
    }
}

impl CachedRenderPipelinePhaseItem for TerrainItem {
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

pub fn extract_terrain_phases(
    mut terrain_phases: ResMut<ViewSortedRenderPhases<TerrainItem>>,
    cameras: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    terrain_phases.clear();

    for (entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }

        terrain_phases.insert(
            RetainedViewEntity {
                main_entity: entity.into(),
                auxiliary_entity: Entity::PLACEHOLDER.into(),
                subview_index: 0,
            },
            default(),
        );
    }
}

#[derive(Component)]
pub struct TerrainViewDepthTexture {
    texture: Texture,
    pub view: TextureView,
    pub depth_view: TextureView,
    pub stencil_view: TextureView,
    /// Depth-copy pipeline specialized for this view's sample count.
    pub copy_pipeline: CachedRenderPipelineId,
    /// Whether the view (and thus this depth texture) is multisampled —
    /// selects the matching depth-copy bind group layout.
    pub multisampled: bool,
}

impl TerrainViewDepthTexture {
    pub fn new(
        texture: CachedTexture,
        copy_pipeline: CachedRenderPipelineId,
        multisampled: bool,
    ) -> Self {
        let depth_view = texture.texture.create_view(&TextureViewDescriptor {
            aspect: TextureAspect::DepthOnly,
            ..default()
        });
        let stencil_view = texture.texture.create_view(&TextureViewDescriptor {
            aspect: TextureAspect::StencilOnly,
            ..default()
        });

        Self {
            texture: texture.texture,
            view: texture.default_view,
            depth_view,
            stencil_view,
            copy_pipeline,
            multisampled,
        }
    }

    pub fn get_attachment(&self) -> RenderPassDepthStencilAttachment<'_> {
        RenderPassDepthStencilAttachment {
            view: &self.view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(0.0), // Clear depth
                store: StoreOp::Store,
            }),
            stencil_ops: Some(Operations {
                load: LoadOp::Clear(0), // Initialize stencil to 0 (lowest priority)
                store: StoreOp::Store,
            }),
        }
    }
}

pub fn prepare_terrain_depth_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    depth_copy_pipeline: Res<DepthCopyPipeline>,
    mut depth_copy_pipelines: ResMut<SpecializedRenderPipelines<DepthCopyPipeline>>,
    gpu_tile_atlases: Res<TerrainComponents<GpuTileAtlas>>,
    default_opaque_renderer_method: Option<Res<DefaultOpaqueRendererMethod>>,
    views_3d: Query<(Entity, &ExtractedCamera, &Msaa, Has<DeferredPrepass>)>,
) {
    // With no terrain, don't hold a full-resolution depth + stencil target
    // per view. Dropping `TerrainViewDepthTexture` also skips `terrain_pass`
    // and `terrain_motion_pass` (both take it in their `ViewQuery`), and not
    // re-fetching the cached texture lets `TextureCache` reclaim it. The
    // motion bind group has to go too — it holds views into that texture and
    // would keep it alive.
    if gpu_tile_atlases.is_empty() {
        for (view, _, _, _) in &views_3d {
            commands
                .entity(view)
                .remove::<TerrainViewDepthTexture>()
                .remove::<TerrainDeferredTargets>()
                .remove::<TerrainMotionBindGroup>();
        }
        return;
    }

    for (view, camera, msaa, has_deferred_prepass) in &views_3d {
        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };

        let samples = msaa.samples();

        let descriptor = TextureDescriptor {
            label: Some("view_depth_texture"),
            size: Extent3d {
                depth_or_array_layers: 1,
                width: physical_target_size.x,
                height: physical_target_size.y,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: TextureDimension::D2,
            format: TERRAIN_DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let cached_texture = texture_cache.get(&device, descriptor);

        // Specialize the depth-copy pass for this view's sample count — the
        // terrain depth texture is multisampled iff the view is, and the
        // copy bind group / pipeline must match (fixes single-sampled DLSS
        // views vs. the old hardcoded 4×).
        let copy_pipeline =
            depth_copy_pipelines.specialize(&pipeline_cache, &depth_copy_pipeline, samples);

        commands.entity(view).insert(TerrainViewDepthTexture::new(
            cached_texture,
            copy_pipeline,
            samples > 1,
        ));

        // Views the terrain renders deferred on additionally get the
        // private G-buffer targets `terrain_deferred_pass` composites into
        // the real prepass attachments (see the component docs) — same
        // gate as every consumer, or the textures would sit resident and
        // unused. A deferred prepass forces `Msaa::Off`, so these are
        // always single-sampled.
        if terrain_renders_deferred(
            has_deferred_prepass,
            default_opaque_renderer_method.as_deref(),
        ) && samples == 1
        {
            let color_descriptor = |label, format| TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_target_size.x,
                    height: physical_target_size.y,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };

            commands.entity(view).insert(TerrainDeferredTargets {
                gbuffer: texture_cache.get(
                    &device,
                    color_descriptor("terrain_deferred_gbuffer", DEFERRED_PREPASS_FORMAT),
                ),
                lighting_pass_id: texture_cache.get(
                    &device,
                    color_descriptor(
                        "terrain_deferred_lighting_pass_id",
                        DEFERRED_LIGHTING_PASS_ID_FORMAT,
                    ),
                ),
            });
        } else {
            commands.entity(view).remove::<TerrainDeferredTargets>();
        }
    }
}

/// Color targets of the terrain's two G-buffer outputs — shared by the
/// deferred terrain pipeline (rendering into the private
/// [`TerrainDeferredTargets`]) and the composite pipeline (restaging them
/// into Bevy's prepass attachments). One definition keeps the formats in
/// lockstep across the two pipelines.
pub(crate) fn deferred_gbuffer_targets() -> [Option<ColorTargetState>; 2] {
    [
        Some(ColorTargetState {
            format: DEFERRED_PREPASS_FORMAT,
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
        Some(ColorTargetState {
            format: DEFERRED_LIGHTING_PASS_ID_FORMAT,
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
    ]
}

/// Private single-sampled copies of the deferred G-buffer attachments the
/// terrain renders into on deferred views. The terrain draw binds Bevy's
/// mesh-view bind group (group 0), which on those views already contains the
/// current-frame deferred texture — a texture can't be both bound and
/// attached in one pass, so the real attachments can't be render targets of
/// the terrain draw itself. [`terrain_deferred_pass`] renders here and then
/// composites into the real attachments in a second, depth-gated fullscreen
/// pass that binds only these private textures.
#[derive(Component)]
pub struct TerrainDeferredTargets {
    pub gbuffer: CachedTexture,
    pub lighting_pass_id: CachedTexture,
}

impl TerrainDeferredTargets {
    /// Attachments into the private targets: they hold only the current
    /// frame's terrain, so they always clear (zero = "no terrain here";
    /// zero also keeps `deferred_lighting_pass_id` at "background" for the
    /// composite).
    fn attachments(&self) -> [Option<RenderPassColorAttachment<'_>>; 2] {
        fn attach(texture: &CachedTexture) -> RenderPassColorAttachment<'_> {
            RenderPassColorAttachment {
                view: &texture.default_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::NONE.into()),
                    store: StoreOp::Store,
                },
            }
        }
        [Some(attach(&self.gbuffer)), Some(attach(&self.lighting_pass_id))]
    }
}

/// Fullscreen pass that copies the terrain phase's depth into the main
/// view depth. Specialized on the view's MSAA sample count so it works
/// for single-sampled (DLSS / `Msaa::Off`) and multisampled views alike —
/// the terrain depth texture's `sample_count` follows the view, and the
/// bind group layout + pipeline + shader must match it.
#[derive(Resource)]
pub struct DepthCopyPipeline {
    vertex: VertexState,
    shader: Handle<Shader>,
}

impl FromWorld for DepthCopyPipeline {
    fn from_world(world: &mut World) -> Self {
        let fullscreen = world.resource::<FullscreenShader>();

        Self {
            vertex: fullscreen.to_vertex_state(),
            shader: world.load_asset(DEPTH_COPY_SHADER),
        }
    }
}

/// Bind group layout for the depth-copy pass — a `texture_depth_2d` that
/// is multisampled to match the view (`terrain_pass` rebuilds the same
/// descriptor to create the bind group).
fn depth_copy_layout(multisampled: bool) -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "depth_copy_pipeline_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (if multisampled {
                texture_depth_2d_multisampled()
            } else {
                texture_depth_2d()
            },),
        ),
    )
}

impl SpecializedRenderPipeline for DepthCopyPipeline {
    /// MSAA sample count of the view.
    type Key = u32;

    fn specialize(&self, samples: u32) -> RenderPipelineDescriptor {
        let multisampled = samples > 1;
        let shader_defs = if multisampled {
            vec!["MULTISAMPLED".into()]
        } else {
            vec![]
        };

        RenderPipelineDescriptor {
            label: Some("depth_copy_pipeline".into()),
            layout: vec![depth_copy_layout(multisampled)],
            immediate_size: 0,
            vertex: self.vertex.clone(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: Some("fragment".into()),
                targets: vec![],
                constants: vec![],
            }),
            primitive: Default::default(),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState {
                count: samples,
                ..Default::default()
            },
            zero_initialize_workgroup_memory: false,
        }
    }
}

pub fn terrain_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
        &TerrainViewDepthTexture,
        Has<DeferredPrepass>,
        Option<&MainPassResolutionOverride>,
    )>,
    mut ctx: RenderContext,
    terrain_phases: Res<ViewSortedRenderPhases<TerrainItem>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    default_opaque_renderer_method: Option<Res<DefaultOpaqueRendererMethod>>,
) {
    let view_entity = view.entity();
    let (
        camera,
        extracted_view,
        target,
        depth,
        terrain_depth,
        has_deferred_prepass,
        resolution_override,
    ) = view.into_inner();

    // On deferred views the terrain renders through `terrain_deferred_pass`
    // instead — skip both the color draw and the `Always` depth overwrite
    // (the phase holds deferred-pipeline items there; see `queue_terrain`).
    if terrain_renders_deferred(
        has_deferred_prepass,
        default_opaque_renderer_method.as_deref(),
    ) {
        return;
    }

    let Some(pipeline) = pipeline_cache.get_render_pipeline(terrain_depth.copy_pipeline) else {
        return;
    };

    let Some(terrain_phase) = terrain_phases.get(&extracted_view.retained_view_entity) else {
        return;
    };

    if terrain_phase.items.is_empty() {
        return;
    }

    // Todo: prepare this in a separate system
    let terrain_depth_view = terrain_depth.texture.create_view(&TextureViewDescriptor {
        aspect: TextureAspect::DepthOnly,
        ..default()
    });
    let depth_layout =
        pipeline_cache.get_bind_group_layout(&depth_copy_layout(terrain_depth.multisampled));
    let depth_copy_bind_group = render_device.create_bind_group(
        None,
        &depth_layout,
        &BindGroupEntries::single(&terrain_depth_view),
    );

    // call this here, otherwise the order between passes is incorrect
    let color_attachments = [Some(target.get_color_attachment())];
    let terrain_depth_stencil_attachment = Some(terrain_depth.get_attachment());
    let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

    {
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("terrain_pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: terrain_depth_stencil_attachment,
            ..default()
        });

        if let Some(viewport) =
            Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
        {
            pass.set_camera_viewport(&viewport);
        }

        terrain_phase.render(&mut pass, world, view_entity).unwrap();
    }

    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            depth_stencil_attachment,
            ..default()
        });
    pass.set_bind_group(0, &depth_copy_bind_group, &[]);
    pass.set_pipeline(pipeline);
    pass.draw(0..3, 0..1);
}

/// Fullscreen pipeline that composites the terrain's private G-buffer
/// targets (see [`TerrainDeferredTargets`]) into Bevy's real deferred
/// prepass attachments. The depth test is the merge: `Greater` (reverse-Z)
/// with depth write, so the terrain's texels and depth land only where the
/// terrain is the closest surface written to the scene depth so far — sky
/// pixels (terrain depth 0) never pass against the cleared depth.
#[derive(Resource)]
pub struct TerrainDeferredCompositePipeline {
    layout: BindGroupLayoutDescriptor,
    id: CachedRenderPipelineId,
}

impl FromWorld for TerrainDeferredCompositePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let fullscreen = world.resource::<FullscreenShader>();

        // Deferred views are always single-sampled (a deferred prepass
        // forces `Msaa::Off`), so unlike the depth-copy pipeline there is
        // no multisampled variant.
        let layout = BindGroupLayoutDescriptor::new(
            "terrain_deferred_composite_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Uint),
                    texture_2d(TextureSampleType::Uint),
                    texture_depth_2d(),
                ),
            ),
        );

        let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("terrain_deferred_composite_pipeline".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: world.load_asset(DEFERRED_COMPOSITE_SHADER),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: deferred_gbuffer_targets().to_vec(),
                constants: vec![],
            }),
            primitive: default(),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Greater),
                stencil: default(),
                bias: default(),
            }),
            multisample: default(),
            zero_initialize_workgroup_memory: false,
        });

        Self { layout, id }
    }
}

/// Deferred-view counterpart of [`terrain_pass`]: renders the terrain phase
/// into the private [`TerrainDeferredTargets`] with the private terrain
/// depth, then composites texels + depth into Bevy's real deferred G-buffer
/// attachments and the scene depth in one depth-gated fullscreen pass, so
/// the deferred lighting pass — or its replacement — shades the terrain
/// like any other deferred geometry. The private indirection exists because
/// the terrain draw's mesh-view bind group contains the current-frame
/// deferred texture on these views (see [`TerrainDeferredTargets`]).
///
/// Scheduled before `early_prepass` (see `plugin.rs`): the composite's
/// `get_attachment()` calls are the frame's first use of the deferred
/// attachments and the scene depth, so they perform the frame's clears and
/// the prepasses load on top; the prepass and deferred meshes depth-test
/// against the merged terrain depth; and the depth copy at the tail of
/// `early_deferred_prepass` restages the scene depth (terrain included)
/// into the prepass depth texture that the lighting passes read.
///
/// KNOWN LIMITATION: motion vectors for the terrain are only written by
/// `terrain_motion_pass`, after the main opaque pass — consumers that read
/// them earlier (e.g. temporal reuse during the lighting pass) see the
/// background's camera-at-infinity motion on terrain pixels and reset their
/// history there while the camera translates. Fixing this requires writing
/// camera-only motion in this pass (previous-view uniforms are not bound in
/// the terrain's group 0) or an early motion pass.
pub fn terrain_deferred_pass(
    world: &World,
    view: ViewQuery<
        (
            &ExtractedCamera,
            &ExtractedView,
            &ViewPrepassTextures,
            &ViewDepthTexture,
            &TerrainViewDepthTexture,
            &TerrainDeferredTargets,
            Option<&MainPassResolutionOverride>,
        ),
        With<DeferredPrepass>,
    >,
    mut ctx: RenderContext,
    terrain_phases: Res<ViewSortedRenderPhases<TerrainItem>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    composite_pipeline: Res<TerrainDeferredCompositePipeline>,
    default_opaque_renderer_method: Option<Res<DefaultOpaqueRendererMethod>>,
) {
    // Same gate as `queue_terrain` (the `With<DeferredPrepass>` filter
    // supplies the view half) — with a forward default, a view with a
    // deferred prepass still renders the terrain forward.
    if !terrain_renders_deferred(true, default_opaque_renderer_method.as_deref()) {
        return;
    }

    let view_entity = view.entity();
    let (
        camera,
        extracted_view,
        prepass_textures,
        depth,
        terrain_depth,
        deferred_targets,
        resolution_override,
    ) = view.into_inner();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(composite_pipeline.id) else {
        return;
    };

    let Some(terrain_phase) = terrain_phases.get(&extracted_view.retained_view_entity) else {
        return;
    };

    if terrain_phase.items.is_empty() {
        return;
    }

    // Both attachments exist on `DeferredPrepass` views.
    let (Some(deferred), Some(deferred_lighting_pass_id)) = (
        &prepass_textures.deferred,
        &prepass_textures.deferred_lighting_pass_id,
    ) else {
        return;
    };

    // Todo: prepare this in a separate system
    let composite_layout = pipeline_cache.get_bind_group_layout(&composite_pipeline.layout);
    let composite_bind_group = render_device.create_bind_group(
        None,
        &composite_layout,
        &BindGroupEntries::sequential((
            &deferred_targets.gbuffer.default_view,
            &deferred_targets.lighting_pass_id.default_view,
            &terrain_depth.depth_view,
        )),
    );

    // call this here, otherwise the order between passes is incorrect
    let private_color_attachments = deferred_targets.attachments();
    let terrain_depth_stencil_attachment = Some(terrain_depth.get_attachment());
    let composite_color_attachments = [
        Some(deferred.get_attachment()),
        Some(deferred_lighting_pass_id.get_attachment()),
    ];
    let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

    let viewport =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override);

    {
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("terrain_deferred_pass"),
            color_attachments: &private_color_attachments,
            depth_stencil_attachment: terrain_depth_stencil_attachment,
            ..default()
        });

        if let Some(viewport) = &viewport {
            pass.set_camera_viewport(viewport);
        }

        terrain_phase.render(&mut pass, world, view_entity).unwrap();
    }

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("terrain_deferred_composite_pass"),
        color_attachments: &composite_color_attachments,
        depth_stencil_attachment,
        ..default()
    });

    if let Some(viewport) = &viewport {
        pass.set_camera_viewport(viewport);
    }

    pass.set_bind_group(0, &composite_bind_group, &[]);
    pass.set_render_pipeline(pipeline);
    pass.draw(0..3, 0..1);
}

// ---------------------------------------------------------------------------
// Terrain motion vectors
//
// The terrain renders in its own phase, so it never enters Bevy's motion
// vector prepass and DLSS/TAA reproject it with zero motion — it ghosts
// while the meshes (which do write motion vectors) stay locked. This pass
// fills that gap: a fullscreen pass that reconstructs each terrain pixel's
// world position from the *rendered* terrain depth and reprojects it
// through the previous frame's view to get the (camera-only) motion
// vector, folding in the CDLOD morph for free. It writes only where the
// terrain is the frontmost surface, so mesh/background motion vectors are
// untouched. Mirrors Bevy's `background_motion_vectors`, restricted to
// terrain depth. Only meaningful with a temporal upscaler, which forces
// `Msaa::Off`, so it is built single-sampled.
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct TerrainMotionPipeline {
    layout: BindGroupLayoutDescriptor,
    id: CachedRenderPipelineId,
}

impl FromWorld for TerrainMotionPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let fullscreen = world.resource::<FullscreenShader>();

        let layout = BindGroupLayoutDescriptor::new(
            "terrain_motion_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<PreviousViewData>(true),
                    texture_depth_2d(),
                ),
            ),
        );

        let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("terrain_motion_pipeline".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: world.load_asset(TERRAIN_MOTION_SHADER),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: MOTION_VECTOR_PREPASS_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                constants: vec![],
            }),
            primitive: default(),
            // Read-only depth test against the final scene depth: the
            // fragment writes the terrain depth as `frag_depth`, so a mesh
            // in front (larger, reversed-Z) fails GreaterEqual and the
            // terrain motion vector is skipped there. Never writes depth.
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: default(),
                bias: default(),
            }),
            multisample: default(),
            zero_initialize_workgroup_memory: false,
        });

        Self { layout, id }
    }
}

/// Per-view bind group for [`terrain_motion_pass`]: the view + previous-view
/// uniforms and the terrain / final-scene depth textures.
#[derive(Component)]
pub struct TerrainMotionBindGroup(BindGroup);

pub fn prepare_terrain_motion_bind_groups(
    mut commands: Commands,
    pipeline: Res<TerrainMotionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    view_uniforms: Res<ViewUniforms>,
    prev_view_uniforms: Res<PreviousViewUniforms>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &TerrainViewDepthTexture)>,
) {
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);

    for (entity, terrain_depth) in &views {
        // Skip bind group creation when the terrain depth
        // texture is multisampled
        if terrain_depth.multisampled {
            commands.entity(entity).remove::<TerrainMotionBindGroup>();
            continue;
        }

        // `binding()` is `None` until the (previous-)view uniforms exist,
        // i.e. only on views with the motion vector prepass (DLSS/TAA).
        let (Some(view_binding), Some(prev_binding)) = (
            view_uniforms.uniforms.binding(),
            prev_view_uniforms.uniforms.binding(),
        ) else {
            continue;
        };

        let bind_group = render_device.create_bind_group(
            "terrain_motion_bind_group",
            &layout,
            &BindGroupEntries::sequential((view_binding, prev_binding, &terrain_depth.depth_view)),
        );

        commands
            .entity(entity)
            .insert(TerrainMotionBindGroup(bind_group));
    }
}

pub fn terrain_motion_pass(
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewPrepassTextures,
        &ViewDepthTexture,
        &TerrainViewDepthTexture,
        &TerrainMotionBindGroup,
        &ViewUniformOffset,
        &PreviousViewUniformOffset,
        Has<DeferredPrepass>,
        Has<TerrainDeferredTargets>,
        Option<&MainPassResolutionOverride>,
    )>,
    mut ctx: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    terrain_motion_pipeline: Res<TerrainMotionPipeline>,
    composite_pipeline: Res<TerrainDeferredCompositePipeline>,
    terrain_phases: Res<ViewSortedRenderPhases<TerrainItem>>,
    default_opaque_renderer_method: Option<Res<DefaultOpaqueRendererMethod>>,
) {
    let (
        camera,
        extracted_view,
        prepass_textures,
        view_depth,
        terrain_depth,
        bind_group,
        view_offset,
        prev_offset,
        has_deferred_prepass,
        has_deferred_targets,
        resolution_override,
    ) = view.into_inner();

    // Mirror the guards of whichever pass rendered the terrain on this view
    // — `terrain_deferred_pass` on deferred views, `terrain_pass` otherwise
    // — exactly. That pass clears and fills the terrain depth this pass
    // samples, and bails on either condition — so relaxing them here would
    // sample an uncleared depth texture (writing motion vectors for last
    // frame's terrain) and restage a depth the terrain never contributed to.
    let terrain_deferred = terrain_renders_deferred(
        has_deferred_prepass,
        default_opaque_renderer_method.as_deref(),
    );
    let copy_pipeline_ready = if terrain_deferred {
        has_deferred_targets
            && pipeline_cache
                .get_render_pipeline(composite_pipeline.id)
                .is_some()
    } else {
        pipeline_cache
            .get_render_pipeline(terrain_depth.copy_pipeline)
            .is_some()
    };
    let drew_terrain = terrain_phases
        .get(&extracted_view.retained_view_entity)
        .is_some_and(|phase| !phase.items.is_empty())
        && copy_pipeline_ready;
    if !drew_terrain {
        return;
    }

    let Some(motion_vectors) = &prepass_textures.motion_vectors else {
        return;
    };

    // DLSS samples depth from the *prepass* depth texture, which Bevy
    // filled (in the prepass node) before the terrain rendered — so it
    // holds mesh-only depth and terrain reads as far-plane/background to
    // DLSS, inconsistent with the terrain color + motion vectors we
    // reproject and therefore rejected from history (this is the residual
    // desync). Restage the final scene depth (terrain + meshes, now in the
    // main view depth) into it before DLSS reads it. Formats match
    // (`CORE_3D_DEPTH_FORMAT`).
    if let Some(prepass_depth) = &prepass_textures.depth {
        ctx.command_encoder().copy_texture_to_texture(
            view_depth.texture.as_image_copy(),
            prepass_depth.texture.texture.as_image_copy(),
            prepass_textures.size,
        );
    }

    let Some(pipeline) = pipeline_cache.get_render_pipeline(terrain_motion_pipeline.id) else {
        return;
    };

    // Load the existing motion vectors (mesh + background) and overwrite
    // only terrain-frontmost pixels — gated by the read-only depth test
    // against the scene depth (never written).
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("terrain_motion_pass"),
        color_attachments: &[Some(motion_vectors.get_attachment())],
        depth_stencil_attachment: Some(view_depth.get_attachment(StoreOp::Store)),
        ..default()
    });

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        pass.set_camera_viewport(&viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group.0, &[view_offset.offset, prev_offset.offset]);
    pass.draw(0..3, 0..1);
}
