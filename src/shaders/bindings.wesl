import bevy_terrain::shaders::types::{TerrainView, Terrain, TileTreeEntry, TileCoordinate, GeometryTile, AttachmentConfig, IndirectBuffer, PrepassState, TerrainShadow};
import bevy_render::view::View;

struct Attachments {
    {0}: AttachmentConfig,
    {1}: AttachmentConfig,
    {2}: AttachmentConfig,
    {3}: AttachmentConfig,
    {4}: AttachmentConfig,
    {5}: AttachmentConfig,
    {6}: AttachmentConfig,
    {7}: AttachmentConfig,
}

struct TerrainBindGroupData {
    terrain: Terrain,
    attachments: Attachments,
}

// refine tiles bindings
@if(PREPASS)
@group(0) @binding(0) var<storage> terrain_view: TerrainView;
@if(PREPASS)
@group(0) @binding(1) var<storage, read_write> approximate_height: f32;
@if(PREPASS)
@group(0) @binding(2) var<storage> tile_tree: array<TileTreeEntry>;
@if(PREPASS)
@group(0) @binding(3) var<storage, read_write> final_tiles: array<GeometryTile>;
@if(PREPASS)
@group(0) @binding(4) var<storage, read_write> temporary_tiles: array<TileCoordinate>;
@if(PREPASS)
@group(0) @binding(5) var<storage, read_write> state: PrepassState;
@if(PREPASS)
@group(2) @binding(0) var<storage, read_write> indirect_buffer: IndirectBuffer;

// terrain shadow map compute bindings
@if(SHADOW_COMPUTE)
@group(0) @binding(0) var<uniform> terrain_shadow: TerrainShadow;
@if(SHADOW_COMPUTE)
@group(2) @binding(0) var<storage> terrain_view: TerrainView;
@if(SHADOW_COMPUTE)
@group(2) @binding(1) var<storage> tile_tree: array<TileTreeEntry>;
@if(SHADOW_COMPUTE)
@group(3) @binding(0) var shadow_map_out: texture_storage_2d<rgba16float, write>;

// terrain view bindings
// Group 1 is reserved for Bevy's view binding-array layout (environment maps, etc.).
@if(!PREPASS && !SHADOW_COMPUTE)
@group(0) @binding(0) var<uniform> view: View;
@if(!PREPASS && !SHADOW_COMPUTE)
@group(3) @binding(0) var<storage> terrain_view: TerrainView;
@if(!PREPASS && !SHADOW_COMPUTE && VERTEX)
@group(3) @binding(1) var<storage> approximate_height: f32;
@if(!PREPASS && !SHADOW_COMPUTE)
@group(3) @binding(2) var<storage> tile_tree: array<TileTreeEntry>;
@if(!PREPASS && !SHADOW_COMPUTE)
@group(3) @binding(3) var<storage> geometry_tiles: array<GeometryTile>;

// terrain shadow map sampling bindings (render pipeline, terrain view bind group)
@if(TERRAIN_SHADOW)
@group(3) @binding(4) var shadow_map: texture_2d<f32>;
@if(TERRAIN_SHADOW)
@group(3) @binding(5) var shadow_map_sampler: sampler;
@if(TERRAIN_SHADOW)
@group(3) @binding(6) var<uniform> terrain_shadow: TerrainShadow;

// terrain bindings
@if(PREPASS)
const TERRAIN_BIND_GROUP: u32 = 1u;
@elif(SHADOW_COMPUTE)
const TERRAIN_BIND_GROUP: u32 = 1u;
@else
const TERRAIN_BIND_GROUP: u32 = 2u;

@group(TERRAIN_BIND_GROUP) @binding(0) var<storage, read> terrain_data: TerrainBindGroupData;
@group(TERRAIN_BIND_GROUP) @binding(1) var terrain_sampler: sampler;
@group(TERRAIN_BIND_GROUP) @binding(2)  var {0}_attachment: texture_2d_array<f32>;
@group(TERRAIN_BIND_GROUP) @binding(3)  var {1}_attachment: texture_2d_array<f32>;
@group(TERRAIN_BIND_GROUP) @binding(4)  var {2}_attachment: texture_2d_array<f32>;
@group(TERRAIN_BIND_GROUP) @binding(5)  var {3}_attachment: texture_2d_array<f32>;
@group(TERRAIN_BIND_GROUP) @binding(6)  var {4}_attachment: texture_2d_array<f32>;
@group(TERRAIN_BIND_GROUP) @binding(7)  var {5}_attachment: texture_2d_array<f32>;
@group(TERRAIN_BIND_GROUP) @binding(8)  var {6}_attachment: texture_2d_array<f32>;
@group(TERRAIN_BIND_GROUP) @binding(9)  var {7}_attachment: texture_2d_array<f32>;
