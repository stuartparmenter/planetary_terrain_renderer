use core::num::NonZeroU64;

use crate::{
    render::{TerrainViewDepthTexture, terrain_pass},
    shaders::PICKING_SHADER,
};
use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{
        core_3d::main_opaque_pass_3d,
        schedule::{Core3d, Core3dSystems},
    },
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    prelude::*,
    render::{
        RenderApp,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{
                storage_buffer_sized, texture_2d_multisampled, texture_depth_2d_multisampled,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        storage::{GpuShaderBuffer, ShaderBuffer},
        sync_component::SyncComponent,
    },
    transform::TransformSystems,
    window::PrimaryWindow,
};
#[cfg(feature = "big_space")]
use big_space::prelude::CellCoord;

#[cfg(feature = "big_space")]
pub fn picking_system(
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform, &CellCoord, &PickingData)>,
) {
    let Ok(window) = window.single() else {
        return;
    };
    let Some(position) = window.cursor_position() else {
        return;
    };
    let cursor_coords = Vec2::new(position.x, window.size().y - position.y) / window.size();

    for (camera, global_transform, &cell, picking_data) in &camera {
        let mut buffer = buffers.get_mut(&picking_data.buffer).unwrap();
        let data = GpuPickingData {
            cursor_coords,
            depth: 0.0,
            stencil: 255,
            world_from_clip: global_transform.to_matrix() * camera.clip_from_view().inverse(),
            cell: IVec3::new(cell.x, cell.y, cell.z),
        };
        buffer.clear();
        buffer.extend_from_slice(&[data]);
    }
}

#[cfg(not(feature = "big_space"))]
pub fn picking_system(
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform, &PickingData)>,
) {
    let Ok(window) = window.single() else {
        return;
    };
    let Some(position) = window.cursor_position() else {
        return;
    };
    let cursor_coords = Vec2::new(position.x, window.size().y - position.y) / window.size();

    for (camera, global_transform, picking_data) in &camera {
        let mut buffer = buffers.get_mut(&picking_data.buffer).unwrap();
        let data = GpuPickingData {
            cursor_coords,
            depth: 0.0,
            stencil: 255,
            world_from_clip: global_transform.to_matrix() * camera.clip_from_view().inverse(),
            cell: IVec3::ZERO,
            _pad: 0,
        };
        buffer.clear();
        buffer.extend_from_slice(&[data]);
    }
}

pub fn picking_readback(on: On<ReadbackComplete>, mut picking_data: Query<&mut PickingData>) {
    let GpuPickingData {
        cursor_coords,
        depth,
        stencil: _stencil,
        world_from_clip,
        cell,
        ..
    } = bytemuck::pod_read_unaligned(&on.event().data[..size_of::<GpuPickingData>()]);

    let ndc_coords = (2.0 * cursor_coords - 1.0).extend(depth);

    let mut picking_data = picking_data.get_mut(on.event().entity).unwrap();
    picking_data.cursor_coords = cursor_coords;
    #[cfg(feature = "big_space")]
    {
        picking_data.cell = CellCoord::new(cell.x, cell.y, cell.z);
    }
    #[cfg(not(feature = "big_space"))]
    {
        picking_data.cell = cell;
    }
    picking_data.translation = (depth > 0.0).then(|| world_from_clip.project_point3(ndc_coords));
    picking_data.world_from_clip = world_from_clip;
}

pub fn picking_hook(mut world: DeferredWorld, context: HookContext) {
    let mut buffers = world.resource_mut::<Assets<ShaderBuffer>>();
    let mut buffer = ShaderBuffer::with_size(
        size_of::<GpuPickingData>() as u64,
        RenderAssetUsages::default(),
    );
    buffer.buffer_usage |= BufferUsages::COPY_SRC;
    let buffer = buffers.add(buffer);

    world
        .commands()
        .entity(context.entity)
        .insert(Readback::buffer(buffer.clone()))
        .observe(picking_readback);

    let mut picking_data = world.get_mut::<PickingData>(context.entity).unwrap();
    picking_data.buffer = buffer;
}

#[derive(Default, Clone, Component)]
#[component(on_add = picking_hook)]
pub struct PickingData {
    pub cursor_coords: Vec2,
    #[cfg(feature = "big_space")]
    pub cell: CellCoord,
    #[cfg(not(feature = "big_space"))]
    pub cell: IVec3,
    pub translation: Option<Vec3>,
    pub world_from_clip: Mat4,
    buffer: Handle<ShaderBuffer>,
}

impl SyncComponent<bevy::render::RenderApp> for PickingData {
    type Target = GpuPickingBuffer;
}

impl ExtractComponent<bevy::render::RenderApp> for PickingData {
    type QueryData = &'static PickingData;
    type QueryFilter = ();
    type Out = GpuPickingBuffer;

    fn extract_component(
        data: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(GpuPickingBuffer(data.buffer.id()))
    }
}

#[derive(Component)]
pub struct GpuPickingBuffer(AssetId<ShaderBuffer>);

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPickingData {
    pub cursor_coords: Vec2,
    pub depth: f32,
    pub stencil: u32,
    pub world_from_clip: Mat4,
    pub cell: IVec3,
    pub _pad: u32,
}

const _: () = assert!(size_of::<GpuPickingData>() == 96);

#[derive(Resource)]
pub struct PickingPipeline {
    id: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
}

impl FromWorld for PickingPipeline {
    fn from_world(world: &mut World) -> Self {
        let _device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let layout = BindGroupLayoutDescriptor::new(
            "picking_pipeline_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_sized(false, NonZeroU64::new(size_of::<GpuPickingData>() as u64)),
                    texture_depth_2d_multisampled(),
                    texture_2d_multisampled(TextureSampleType::Uint),
                ),
            ),
        );

        let id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: world.load_asset(PICKING_SHADER),
            shader_defs: vec![],
            entry_point: Some("pick".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });

        Self { id, layout }
    }
}

pub fn picking_pass(
    view: ViewQuery<(&GpuPickingBuffer, &TerrainViewDepthTexture)>,
    mut ctx: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    picking_pipeline: Res<PickingPipeline>,
    buffer: Res<RenderAssets<GpuShaderBuffer>>,
) {
    let (picking_buffer, depth) = view.into_inner();

    let Some(pipeline) = pipeline_cache.get_compute_pipeline(picking_pipeline.id) else {
        return;
    };

    let Some(buffer) = buffer.get(picking_buffer.0) else {
        return;
    };

    let layout = pipeline_cache.get_bind_group_layout(&picking_pipeline.layout);
    let bind_group = ctx.render_device().create_bind_group(
        None,
        &layout,
        &BindGroupEntries::sequential((
            buffer.buffer.as_entire_binding(),
            &depth.depth_view,
            &depth.stencil_view,
        )),
    );

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());
    pass.set_bind_group(0, &bind_group, &[]);
    pass.set_pipeline(pipeline);
    pass.dispatch_workgroups(1, 1, 1);
}

pub struct TerrainPickingPlugin;

impl Plugin for TerrainPickingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            picking_system.after(TransformSystems::Propagate),
        )
        .add_plugins(ExtractComponentPlugin::<PickingData>::default());

        app.sub_app_mut(RenderApp).add_systems(
            Core3d,
            picking_pass
                .after(terrain_pass)
                .before(main_opaque_pass_3d)
                .in_set(Core3dSystems::MainPass),
        );
    }
    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<PickingPipeline>();
    }
}
