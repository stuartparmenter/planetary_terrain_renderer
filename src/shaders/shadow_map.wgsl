// Generates the RDR2-style terrain shadow map: for every texel of a small top-down
// map centered on the view, the terrain height map is raymarched towards the sun and
// the highest occluding elevation (R) plus the ray length to that occluder (G) are
// stored. At shading time fragments compare their elevation against R and use G to
// widen the penumbra for cheap soft shadows.

#import bevy_terrain::bindings::{terrain_shadow, shadow_map_out}
#import bevy_terrain::heightfield::sample_height_at_world

// Elevation sentinel meaning "no occluder" (well below any terrain, fits rg16float).
const NO_OCCLUDER: f32 = -60000.0;

@compute @workgroup_size(8, 8, 1)
fn compute_shadow_map(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let size = u32(terrain_shadow.map_size);
    if (invocation.x >= size || invocation.y >= size) { return; }

    // Horizontal offset of this texel from the map center, in meters.
    let uv = (vec2<f32>(invocation.xy) + 0.5) / terrain_shadow.map_size;
    let origin = (uv * 2.0 - 1.0) * terrain_shadow.extent;

    // Decompose the sun direction into the local tangent frame of the map center.
    let up        = terrain_shadow.up;
    let sun_up    = dot(terrain_shadow.light_direction, up);
    let horizon   = terrain_shadow.light_direction - sun_up * up;
    let horizon_length = length(horizon);

    var intersection_height = NO_OCCLUDER;
    var ray_length          = terrain_shadow.max_distance;

    if (horizon_length > 1e-4) {
        let slope = sun_up / horizon_length;
        let direction = vec2<f32>(dot(horizon, terrain_shadow.east),
                                  dot(horizon, terrain_shadow.north)) / horizon_length;

        // Exponentially growing steps out to the max scan distance. Start beyond the
        // texel's own cell, so the coarse height there cannot self-shadow fragments
        // that sit below the cell average (close-range shadows are handled by the
        // regular shadow maps).
        let start  = 1.5 * terrain_shadow.texel_size;
        let count  = terrain_shadow.steps;
        let growth = pow(terrain_shadow.max_distance / start, 1.0 / f32(count - 1u));

        var t = start;
        for (var i = 0u; i < count; i = i + 1u) {
            let sample_xy = origin + t * direction;
            let world_position = terrain_shadow.center
                + sample_xy.x * terrain_shadow.east
                + sample_xy.y * terrain_shadow.north;

            let height = sample_height_at_world(world_position, terrain_shadow.unit_from_world,
                                                terrain_shadow.lod);

            // Elevation below which a point at this texel is occluded by the sample:
            // terrain height, minus the spherical curvature drop, minus the elevation
            // the sun ray gains while traveling the horizontal distance t.
            let occlusion_height = height - t * t * terrain_shadow.curvature - t * slope;

            if (occlusion_height > intersection_height) {
                intersection_height = occlusion_height;
                ray_length          = t;
            }

            t = t * growth;
        }
    } else if (sun_up < 0.0) {
        // Sun pointing straight down: everything is occluded.
        intersection_height = 60000.0;
        ray_length          = terrain_shadow.texel_size;
    }

    // Clamp both channels into rg16float's finite range (+-65504): with the
    // light below the horizon the accumulated occlusion height grows past it,
    // and `ray_length` starts at `max_distance`, which the settings allow
    // beyond 65 km. An out-of-range store converts to Inf on some drivers,
    // which the decode would turn into NaN. +-60000 decodes identically:
    // R = -60000 is the no-occluder sentinel, +60000 is fully occluded, and a
    // 60 km ray length already exceeds any visible penumbra width.
    textureStore(shadow_map_out, vec2<i32>(invocation.xy),
                 vec4<f32>(clamp(intersection_height, NO_OCCLUDER, 60000.0),
                           min(ray_length, 60000.0), 0.0, 0.0));
}
