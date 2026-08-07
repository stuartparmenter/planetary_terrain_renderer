import bevy_terrain::shaders::types::{Coordinate, Blend, INVALID_ATLAS_INDEX};
import bevy_terrain::shaders::bindings::terrain_data;
import bevy_terrain::shaders::attachments::{sample_height, sample_height_mask};
import bevy_terrain::shaders::functions::lookup_tile;

const HEIGHTFIELD_SIGMA: f32 = 0.87 * 0.87;

// Transforms a world position into terrain unit space (the unit cube sphere for
// spherical terrains, or the unit plane for planar ones).
fn world_to_unit_position(world_position: vec3<f32>, unit_from_world: mat4x4<f32>) -> vec3<f32> {
    let unit = (unit_from_world * vec4<f32>(world_position, 1.0)).xyz;
    @if(SPHERICAL) {
        return normalize(unit);
    } @else {
        return unit;
    }
}

// Inverse of the cube-sphere projection in `Coordinate::from_unit_position`.
fn unit_to_coordinate(unit: vec3<f32>) -> Coordinate {
    @if(SPHERICAL) {
    let abs_u = abs(unit);
    var face: u32;
    var xy: vec2<f32>;

    if (abs_u.x >= abs_u.y && abs_u.x >= abs_u.z) {
        if (unit.x < 0.0) {
            face = 0u;
            xy = vec2(unit.z, -unit.y) / abs_u.x;
        } else {
            face = 3u;
            xy = vec2(-unit.y, unit.z) / abs_u.x;
        }
    } else if (abs_u.y >= abs_u.x && abs_u.y >= abs_u.z) {
        if (unit.y < 0.0) {
            face = 5u;
            xy = vec2(unit.z, unit.x) / abs_u.y;
        } else {
            face = 2u;
            xy = vec2(unit.x, unit.z) / abs_u.y;
        }
    } else if (unit.z < 0.0) {
        face = 4u;
        xy = vec2(-unit.y, unit.x) / abs_u.z;
    } else {
        face = 1u;
        xy = vec2(unit.x, -unit.y) / abs_u.z;
    }

    let uv = clamp(
        0.5 * xy * sqrt((1.0 + HEIGHTFIELD_SIGMA) / (1.0 + HEIGHTFIELD_SIGMA * xy * xy)) + 0.5,
        vec2(0.0),
        vec2(1.0),
    );

    return Coordinate(face, 0u, vec2<u32>(0u), uv);
    } @else {
    let uv = clamp(unit.xz + 0.5, vec2(0.0), vec2(1.0));

    return Coordinate(0u, 0u, vec2<u32>(0u), uv);
    }
}

// Samples the terrain height (meters above the ellipsoid) at an arbitrary world position.
//
// A single fixed `lod` must be used for all samples of a pass. It has to be coarse
// enough that the (viewer-centered) tile tree window at that lod covers every sampled
// position; otherwise the wrapping tree lookup would return unrelated tiles.
fn sample_height_at_world(world_position: vec3<f32>, unit_from_world: mat4x4<f32>, lod: u32) -> f32 {
    let unit       = world_to_unit_position(world_position, unit_from_world);
    let coordinate = unit_to_coordinate(unit);
    let tile       = lookup_tile(coordinate, Blend(lod, 0.0));

    // Outside the dataset (no tile at any lod, or masked-out texels of an edge
    // tile) there is no terrain: report the lowest height instead of sampling
    // garbage, so the void never casts shadows.
    if (tile.index == INVALID_ATLAS_INDEX || sample_height_mask(tile)) {
        return terrain_data.terrain.min_height;
    }

    return sample_height(tile);
}
