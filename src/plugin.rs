use crate::{
    floating_origin::register_plugins,
    formats::{TerrainConfigLoader, TiffLoader},
    preprocess::{MipPipelines, mip_prepass},
    render::{
        DepthCopyPipeline, GpuTerrain, GpuTerrainShadow, GpuTerrainView,
        TerrainDeferredCompositePipeline, TerrainItem, TerrainMotionPipeline,
        TerrainShadowPipelines, TerrainTilingPrepassPipelines, TerrainUniform, TilingPrepassItem,
        extract_terrain_phases, extract_terrain_uniform, prepare_terrain_composite_bind_groups,
        prepare_terrain_depth_textures, prepare_terrain_motion_bind_groups, queue_tiling_prepass,
        terrain_deferred_pass, terrain_motion_pass, terrain_pass, terrain_shadow_pass,
        tiling_prepass,
    },
    shaders::{InternalShaders, load_terrain_shaders},
    terrain::{TerrainComponents, TerrainConfig},
    terrain_data::{
        AttachmentLabel, GpuTileAtlas, TerrainTileDropped, TerrainTileReady, TerrainViewKey,
        TileAtlas, TileTree, finish_loading, start_loading,
    },
    terrain_shadow::{
        TerrainShadowSettings, TerrainShadowUniform, extract_terrain_shadow, update_terrain_shadow,
    },
    terrain_view::TerrainViewComponents,
};
use bevy::ecs::entity::EntityHashSet;
use bevy::transform::TransformSystems;
use bevy::{
    core_pipeline::{
        core_3d::main_opaque_pass_3d,
        prepass::node::early_prepass,
        schedule::{Core3d, Core3dSystems, camera_driver},
    },
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        render_phase::{DrawFunctions, ViewSortedRenderPhases, sort_phase_system},
        render_resource::*,
        renderer::RenderGraph,
    },
};

#[derive(Resource)]
pub struct TerrainSettings {
    pub attachments: Vec<AttachmentLabel>,
    pub atlas_size: u32,
}

impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            attachments: vec![AttachmentLabel::Height],
            atlas_size: 1028,
        }
    }
}

impl TerrainSettings {
    pub fn new(custom_attachments: Vec<&str>) -> Self {
        let mut attachments = vec![AttachmentLabel::Height];
        attachments.extend(
            custom_attachments
                .into_iter()
                .map(|name| AttachmentLabel::Custom(name.to_string())),
        );

        Self {
            attachments,
            atlas_size: 1028,
        }
    }
}

/// Drop the per-terrain state of terrains whose entity has been despawned.
/// Several resources are keyed by terrain entity and are never otherwise
/// pruned, so without this a despawned terrain leaks its `TileTree` and
/// `TileAtlas::update` panics looking the entity up.
fn cleanup_despawned_terrains(
    mut commands: Commands,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    mut shadow_uniforms: ResMut<TerrainViewComponents<TerrainShadowUniform>>,
    terrains: Query<Entity, With<TileAtlas>>,
    readbacks: Query<(Entity, &TerrainViewKey)>,
) {
    let live: EntityHashSet = terrains.iter().collect();
    tile_trees.retain(|(terrain, _view), _| live.contains(terrain));
    shadow_uniforms.retain(|(terrain, _view), _| live.contains(terrain));

    // `TileTree::new` spawns a standalone `Readback` entity per tile tree.
    // It is not a child of the terrain, so despawning the terrain leaves it
    // reading back a buffer for a tile tree that no longer exists.
    for (entity, &TerrainViewKey((terrain, _view))) in &readbacks {
        if !live.contains(&terrain) {
            commands.entity(entity).despawn();
        }
    }
}

/// Render-world half of [`cleanup_despawned_terrains`]. Runs before the
/// `initialize` systems so a terrain respawned in the same frame does not
/// inherit the old entity's GPU state, and so the terrain phase stops
/// drawing a terrain whose entity is gone.
fn cleanup_despawned_gpu_terrains(
    mut gpu_tile_atlases: ResMut<TerrainComponents<GpuTileAtlas>>,
    mut gpu_terrains: ResMut<TerrainComponents<GpuTerrain>>,
    mut terrain_uniforms: ResMut<TerrainComponents<TerrainUniform>>,
    mut gpu_terrain_views: ResMut<TerrainViewComponents<GpuTerrainView>>,
    mut gpu_terrain_shadows: ResMut<TerrainViewComponents<GpuTerrainShadow>>,
    mut shadow_uniforms: ResMut<TerrainViewComponents<TerrainShadowUniform>>,
    mut tiling_prepass_items: ResMut<TerrainViewComponents<TilingPrepassItem>>,
    terrains: Extract<Query<Entity, With<TileAtlas>>>,
) {
    let live: EntityHashSet = terrains.iter().collect();
    gpu_tile_atlases.retain(|terrain, _| live.contains(terrain));
    gpu_terrains.retain(|terrain, _| live.contains(terrain));
    terrain_uniforms.retain(|terrain, _| live.contains(terrain));
    gpu_terrain_views.retain(|(terrain, _view), _| live.contains(terrain));
    gpu_terrain_shadows.retain(|(terrain, _view), _| live.contains(terrain));
    shadow_uniforms.retain(|(terrain, _view), _| live.contains(terrain));
    tiling_prepass_items.retain(|(terrain, _view), _| live.contains(terrain));
}

/// The plugin for the terrain renderer.
pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        register_plugins(app);

        app.init_asset::<TerrainConfig>()
            .init_resource::<InternalShaders>()
            .init_resource::<TerrainViewComponents<TileTree>>()
            .init_resource::<TerrainSettings>()
            .init_resource::<TerrainShadowSettings>()
            .init_resource::<TerrainViewComponents<TerrainShadowUniform>>()
            .init_asset_loader::<TerrainConfigLoader>()
            .init_asset_loader::<TiffLoader>()
            .add_message::<TerrainTileReady>()
            .add_message::<TerrainTileDropped>()
            .add_systems(
                PostUpdate,
                (
                    // Todo: enable visibility checking again
                    // check_visibility::<With<TileAtlas>>.in_set(VisibilitySystems::CheckVisibility),
                    (
                        cleanup_despawned_terrains,
                        TileTree::compute_requests,
                        finish_loading,
                        TileAtlas::stage_uploads,
                        TileAtlas::update,
                        TileAtlas::emit_tile_events,
                        start_loading,
                        TileTree::adjust_to_tile_atlas,
                        TileTree::generate_surface_approximation,
                        TileTree::update_terrain_view_buffer,
                        update_terrain_shadow,
                    )
                        .chain()
                        .after(TransformSystems::Propagate),
                ),
            );
        app.sub_app_mut(RenderApp)
            .init_resource::<SpecializedComputePipelines<MipPipelines>>()
            .init_resource::<SpecializedComputePipelines<TerrainTilingPrepassPipelines>>()
            .init_resource::<SpecializedComputePipelines<TerrainShadowPipelines>>()
            .init_resource::<TerrainComponents<GpuTileAtlas>>()
            .init_resource::<TerrainComponents<GpuTerrain>>()
            .init_resource::<TerrainComponents<TerrainUniform>>()
            .init_resource::<TerrainViewComponents<GpuTerrainView>>()
            .init_resource::<TerrainViewComponents<GpuTerrainShadow>>()
            .init_resource::<TerrainViewComponents<TilingPrepassItem>>()
            .init_resource::<TerrainShadowSettings>()
            .init_resource::<TerrainViewComponents<TerrainShadowUniform>>()
            .init_resource::<DrawFunctions<TerrainItem>>()
            .init_resource::<ViewSortedRenderPhases<TerrainItem>>()
            .add_systems(
                ExtractSchedule,
                (
                    extract_terrain_phases,
                    extract_terrain_shadow,
                    extract_terrain_uniform,
                    GpuTileAtlas::initialize,
                    GpuTileAtlas::extract.after(GpuTileAtlas::initialize),
                    GpuTerrain::initialize.after(GpuTileAtlas::initialize),
                    GpuTerrainView::initialize,
                    GpuTerrainShadow::initialize,
                )
                    .after(cleanup_despawned_gpu_terrains),
            )
            .add_systems(ExtractSchedule, cleanup_despawned_gpu_terrains)
            .add_systems(
                Render,
                (
                    (
                        GpuTileAtlas::prepare,
                        GpuTerrain::prepare,
                        GpuTerrainShadow::prepare,
                        GpuTerrainView::prepare_terrain_view,
                        GpuTerrainView::prepare_indirect,
                        GpuTerrainView::prepare_refine_tiles,
                    )
                        .in_set(RenderSystems::PrepareBindGroups),
                    sort_phase_system::<TerrainItem>.in_set(RenderSystems::PhaseSort),
                    prepare_terrain_depth_textures.in_set(RenderSystems::PrepareResources),
                    (
                        prepare_terrain_motion_bind_groups,
                        prepare_terrain_composite_bind_groups,
                    )
                        .in_set(RenderSystems::PrepareBindGroups),
                    (
                        queue_tiling_prepass,
                        GpuTileAtlas::queue,
                        GpuTerrainShadow::queue,
                    )
                        .in_set(RenderSystems::Queue),
                    GpuTileAtlas::_cleanup.in_set(RenderSystems::Cleanup),
                ),
            )
            .add_systems(
                RenderGraph,
                (mip_prepass, tiling_prepass, terrain_shadow_pass)
                    .chain()
                    .before(camera_driver),
            )
            .add_systems(
                Core3d,
                (
                    // Must run before `early_deferred_prepass` (its deferred
                    // meshes depth-test against the terrain, and the depth
                    // copy at its tail must include it), which is not
                    // publicly nameable — so run before `early_prepass`,
                    // the head of the prepass chain. The attachments'
                    // clear-on-first-use semantics make this equivalent:
                    // this pass performs the frame's clears, the prepasses
                    // load, and the `Greater` depth merge composes the same
                    // either way.
                    terrain_deferred_pass
                        .before(early_prepass)
                        .in_set(Core3dSystems::Prepass),
                    (
                        terrain_pass.before(main_opaque_pass_3d),
                        // Runs after the opaque meshes so the final scene depth
                        // is complete — it masks terrain motion vectors against
                        // it. Still before the temporal upscaler consumes them.
                        terrain_motion_pass.after(main_opaque_pass_3d),
                    )
                        .in_set(Core3dSystems::MainPass),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let attachments = app
            .world()
            .resource::<TerrainSettings>()
            .attachments
            .clone();

        load_terrain_shaders(app, &attachments);

        app.sub_app_mut(RenderApp)
            .init_resource::<TerrainTilingPrepassPipelines>()
            .init_resource::<TerrainShadowPipelines>()
            .init_resource::<MipPipelines>()
            .init_resource::<DepthCopyPipeline>()
            .init_resource::<SpecializedRenderPipelines<DepthCopyPipeline>>()
            .init_resource::<TerrainDeferredCompositePipeline>()
            .init_resource::<TerrainMotionPipeline>();
    }
}
