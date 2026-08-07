import bevy_terrain::shaders::types::{Blend, Coordinate, WorldCoordinate, AtlasTile, TangentSpace};
import bevy_terrain::shaders::bindings::{terrain_data, terrain_view, geometry_tiles};
@if(TERRAIN_SHADOW)
import bevy_terrain::shaders::bindings::{shadow_map, shadow_map_sampler, terrain_shadow};
@if(TERRAIN_SHADOW)
import bevy_terrain::shaders::terrain_shadow::terrain_shadow_factor_at;
import bevy_terrain::shaders::functions::{compute_coordinate, compute_world_coordinate, compute_blend, compute_tangent_space, lookup_tile, apply_height, high_precision};
import bevy_terrain::shaders::attachments::{sample_height_mask, sample_surface_gradient};
import bevy_terrain::shaders::debug::{show_data_lod, show_geometry_lod, show_tile_tree, show_pixels};
import bevy_pbr::render::pbr_types::{PbrInput, pbr_input_new};
import bevy_pbr::render::pbr_functions::{calculate_view, apply_pbr_lighting};
import bevy_pbr::render::mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT;
@if(DEFERRED_PREPASS)
import bevy_pbr::deferred::functions::deferred_gbuffer_from_pbr_input;

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tile_uv: vec2<f32>,
    @location(1) @interpolate(flat) tile_index: u32,
    @location(2) view_distance: f32,
    @location(3) height: f32,
}

// Under DEFERRED_PREPASS the fragment writes Bevy's packed G-buffer and the
// lighting-pass id instead of a lit color; the locations match the deferred
// terrain pipeline's two color targets.
@if(DEFERRED_PREPASS)
struct FragmentOutput {
    @location(0) deferred: vec4<u32>,
    @location(1) deferred_lighting_pass_id: u32,
}
@if(!DEFERRED_PREPASS)
struct FragmentOutput {
    @location(0) color: vec4<f32>
}

struct FragmentInfo {
    clip_position: vec4<f32>,
    tile_index: u32,
    height: f32,
    coordinate: Coordinate,
    world_coordinate: WorldCoordinate,
    tangent_space: TangentSpace,
    blend: Blend,
}

fn fragment_info(input: FragmentInput) -> FragmentInfo{
    var info: FragmentInfo;
    info.clip_position    = input.clip_position;
    info.tile_index       = input.tile_index;
    info.height           = input.height;
    info.coordinate       = compute_coordinate(input.tile_index, input.tile_uv);
    info.world_coordinate = compute_world_coordinate(info.coordinate, input.height, input.view_distance);
    info.tangent_space    = compute_tangent_space(info.world_coordinate);
    info.blend            = compute_blend(info.world_coordinate.view_distance);
    return info;
}

// Samples the RDR2-style terrain shadow map through the terrain view bind
// group; the decode itself lives in `bevy_terrain::shaders::terrain_shadow`.
@if(TERRAIN_SHADOW)
fn terrain_shadow_factor(world_position: vec3<f32>) -> f32 {
    return terrain_shadow_factor_at(world_position, terrain_shadow, shadow_map, shadow_map_sampler);
}

// Applies the map to a `PbrInput`: the factor paired with the direction of
// the light it was marched for, which scopes it to that light inside
// `apply_pbr_lighting`.
@if(TERRAIN_SHADOW)
fn apply_terrain_shadow(pbr_input: ptr<function, PbrInput>, world_position: vec3<f32>) {
    (*pbr_input).directional_shadow_factor = terrain_shadow_factor(world_position);
    (*pbr_input).directional_shadow_direction = terrain_shadow.light_direction;
}

fn fragment_output(info: ptr<function, FragmentInfo>, output: ptr<function, FragmentOutput>, color: vec4<f32>, surface_gradient: vec3<f32>) {
    let world_position = vec4<f32>(apply_height((*info).world_coordinate, (*info).height), 1.0);

// DEFERRED_PREPASS implies LIGHTING (enforced in `queue_terrain`): the
// G-buffer output has no color slot for the unlit path, and the packed
// PbrInput comes from the shared assembly below.
    @if(LIGHTING) {
        var pbr_input: PbrInput                 = pbr_input_new();
        pbr_input.material.base_color           = color;
        pbr_input.material.perceptual_roughness = 1.0;
        pbr_input.material.reflectance          = vec3<f32>(0.0);
        pbr_input.frag_coord                    = (*info).clip_position;
        pbr_input.world_position                = world_position;
        pbr_input.world_normal                  = (*info).world_coordinate.normal;
        pbr_input.N                             = normalize((*info).world_coordinate.normal - surface_gradient);
        // Receive Bevy cascade / contact shadows from mesh casters (spheres, etc.).
        pbr_input.flags                         = MESH_FLAGS_SHADOW_RECEIVER_BIT;

        @if(DEFERRED_PREPASS) {
            // Pack into the G-buffer for the deferred lighting pass (or its
            // replacement) to shade. The packer applies the gamma / rgb9e5
            // encodings itself and never reads `V`; the terrain-shadow factor has
            // no G-buffer slot and is skipped.
            (*output).deferred                  = deferred_gbuffer_from_pbr_input(pbr_input);
            // 1 = PBR deferred lighting (`pbr_input_new` default).
            (*output).deferred_lighting_pass_id = pbr_input.material.deferred_lighting_pass_id;
        } @else {
            pbr_input.V = calculate_view(world_position, pbr_input.is_orthographic);
            @if(TERRAIN_SHADOW)
            apply_terrain_shadow(&pbr_input, world_position.xyz);

            (*output).color = apply_pbr_lighting(pbr_input);
        }
    } @else {
        (*output).color = color;
    }
}

// Debug overlays write lit colors — they only exist on the forward path
// (`FragmentOutput` has no color slot under DEFERRED_PREPASS).
@if(!DEFERRED_PREPASS)
fn fragment_debug(info: ptr<function, FragmentInfo>, output: ptr<function, FragmentOutput>, tile: AtlasTile, surface_gradient: vec3<f32>) {
    let normal = normalize((*info).world_coordinate.normal - surface_gradient);

    @if(SHOW_DATA_LOD) {
        (*output).color = show_data_lod((*info).blend, tile);
    }
    @if(SHOW_GEOMETRY_LOD) {
        (*output).color = show_geometry_lod((*info).coordinate, (*info).tile_index);
    }
    @if(SHOW_TILE_TREE) {
        (*output).color = show_tile_tree((*info).coordinate, (*info).world_coordinate);
    }
    @if(SHOW_PIXELS) {
        (*output).color = mix((*output).color, show_pixels(tile), 0.5);
    }
    @if(SHOW_UV) {
        (*output).color = vec4<f32>(tile.coordinate.uv, 0.0, 1.0);
    }
    @if(SHOW_NORMALS) {
        (*output).color = vec4<f32>(normal, 1.0);
    }

    @if(TEST3)
    if (high_precision((*info).world_coordinate.view_distance)) {
        (*output).color = mix((*output).color, vec4<f32>(0.3), 0.5);
    }
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var info = fragment_info(input);

    let tile             = lookup_tile(info.coordinate, info.blend);
    let mask             = sample_height_mask(tile);
    let color            = vec4<f32>(0.5);
    let surface_gradient = sample_surface_gradient(tile, info.tangent_space);

    if (mask) { discard; }

    var output: FragmentOutput;
    fragment_output(&info, &output, color, surface_gradient);
    @if(DEFERRED_PREPASS) {
        return output;
    } @else {
        fragment_debug(&info, &output, tile, surface_gradient);
        return FragmentOutput(vec4<f32>(output.color.xyz, 1.0));
    }
}
