use crate::{
    math::{TerrainShape, TileCoordinate},
    plugin::TerrainSettings,
    terrain::TerrainConfig,
    terrain_data::{
        Attachment, AttachmentData, AttachmentLabel, AttachmentTile, AttachmentTileWithData,
        DefaultLoader, TerrainTileDropped, TerrainTileReady, TileTree, TileTreeEntry,
    },
    terrain_view::TerrainViewComponents,
};
use bevy::{
    camera::visibility::{VisibilityClass, add_visibility_class},
    platform::collections::{HashMap, HashSet},
    prelude::*,
    tasks::Task,
};
#[cfg(feature = "big_space")]
use big_space::prelude::CellCoord;
use std::collections::VecDeque;

/// Upper bound on attachment bytes staged for GPU upload per frame and atlas.
/// This bounds the `write_texture` cost of `GpuTileAtlas::prepare` — draining
/// a whole loader burst in one frame stalls it for several milliseconds.
const UPLOAD_BUDGET_BYTES: usize = 8 * 1024 * 1024;

/// The current state of a tile of a [`TileAtlas`].
///
/// This indicates, whether the tile is loading or loaded and ready to be used.
#[derive(Clone, Copy, Debug)]
enum LoadingState {
    /// The tile is loading, but can not be used yet.
    Loading(u32),
    /// The tile is loaded and can be used.
    Loaded,
}

/// The internal representation of a present tile in a [`TileAtlas`].
struct TileState {
    /// Indicates whether or not the tile is loading or loaded.
    state: LoadingState,
    /// The index of the tile inside the atlas.
    atlas_index: u32,
    /// The count of [`TileTrees`] that have requested this tile.
    requests: u32,
    /// Identifies the `request_tile` insert that created this state. Issued
    /// loads carry this generation, and only matching results may decrement
    /// the `Loading` counter — a coordinate can be evicted and re-requested
    /// while loads for its predecessor are still in flight or queued.
    generation: u32,
}

// Todo: rename to terrain?
// Todo: consider turning this into an asset

/// A sparse storage of all terrain attachments, which streams data in and out of memory
/// depending on the decisions of the corresponding [`TileTree`]s.
///
/// A tile is considered present and assigned an [`u32`] as soon as it is
/// requested by any tile_tree. Then the tile atlas will start loading all of its attachments
/// by storing the [`TileCoordinate`] (for one frame) in `load_events` for which
/// attachment-loading-systems can listen.
/// Tiles that are not being used by any tile_tree anymore are cached (LRU),
/// until new atlas indices are required.
///
/// The [`u32`] can be used for accessing the attached data in systems by the CPU
/// and in shaders by the GPU.
#[derive(Component)]
#[require(Transform, Visibility, VisibilityClass, DefaultLoader)]
#[cfg_attr(feature = "big_space", require(CellCoord))]
#[component(on_add = add_visibility_class::<TileAtlas>)]
pub struct TileAtlas {
    pub(crate) attachments: HashMap<AttachmentLabel, Attachment>, // stores the attachment data
    tile_states: HashMap<TileCoordinate, TileState>,
    unused_indices: VecDeque<u32>,
    existing_tiles: HashSet<TileCoordinate>,
    pub(crate) uploading_tiles: Vec<AttachmentTileWithData>,
    pub(crate) downloading_tiles: Vec<Task<AttachmentTileWithData>>,
    pub(crate) to_load: Vec<AttachmentTile>,
    /// Attachments that finished loading, waiting on [`Self::stage_uploads`]'s
    /// per-frame byte budget before entering `uploading_tiles`.
    pending_uploads: VecDeque<(AttachmentTile, AttachmentData)>,
    /// Tiles that became resident this update, drained into [`TerrainTileReady`] messages.
    ready_tiles: Vec<(TileCoordinate, u32)>,
    /// Ready tiles that were evicted this update, drained into [`TerrainTileDropped`] messages.
    dropped_tiles: Vec<(TileCoordinate, u32)>,
    /// Generation of the most recent tile-state insert (see [`TileState::generation`]).
    generation: u32,

    pub(crate) lod_count: u32,
    pub(crate) min_height: f32,
    pub(crate) max_height: f32,
    pub(crate) height_scale: f32,
    pub(crate) shape: TerrainShape,
}

impl TileAtlas {
    /// Creates a new tile_tree from a terrain config.
    pub fn new(config: &TerrainConfig, settings: &TerrainSettings) -> Self {
        let attachments = config
            .attachments
            .iter()
            .map(|(label, attachment)| (label.clone(), Attachment::new(attachment, &config.path)))
            .collect();

        Self {
            attachments,
            tile_states: default(),
            unused_indices: (0..settings.atlas_size).collect(),
            existing_tiles: HashSet::from_iter(config.tiles.clone()),
            to_load: default(),
            uploading_tiles: default(),
            downloading_tiles: default(),
            pending_uploads: default(),
            ready_tiles: default(),
            dropped_tiles: default(),
            generation: 0,
            lod_count: config.lod_count,
            min_height: config.min_height,
            max_height: config.max_height,
            height_scale: 1.0,
            shape: config.shape,
        }
    }

    pub(crate) fn get_best_tile(&self, tile_coordinate: TileCoordinate) -> TileTreeEntry {
        let mut best_tile_coordinate = tile_coordinate;

        if !self.existing_tiles.contains(&tile_coordinate) {
            return TileTreeEntry::default();
        }

        loop {
            if best_tile_coordinate == TileCoordinate::INVALID {
                // highest lod is not loaded
                return TileTreeEntry::default();
            }

            if let Some(tile) = self.tile_states.get(&best_tile_coordinate) {
                if matches!(tile.state, LoadingState::Loaded) {
                    // found best loaded tile
                    return TileTreeEntry {
                        atlas_index: tile.atlas_index,
                        atlas_lod: best_tile_coordinate.lod,
                    };
                }
            }

            best_tile_coordinate = best_tile_coordinate
                .parent()
                .unwrap_or(TileCoordinate::INVALID);
        }
    }

    pub(crate) fn tile_loaded(&mut self, tile: AttachmentTile, data: AttachmentData) {
        // A load that outlived its tile state — the slot was evicted, and
        // possibly re-requested under a newer generation — must not enter the
        // queue: it belongs to the evicted predecessor, not the state now at
        // this coordinate.
        if !self.tile_generation_matches(&tile) {
            return;
        }

        // Deferred to `stage_uploads`: the tile's `Loaded` flip — which makes
        // `get_best_tile` serve it to the tile trees and fires
        // [`TerrainTileReady`] — has to stay on the frame its data enters
        // `uploading_tiles`, or the shader would sample the slot before the
        // texels arrive.
        self.pending_uploads.push_back((tile, data));
    }

    fn tile_generation_matches(&self, tile: &AttachmentTile) -> bool {
        self.tile_states
            .get(&tile.coordinate)
            .is_some_and(|tile_state| tile_state.generation == tile.generation)
    }

    /// Stages loaded attachments for GPU upload, draining the pending queue
    /// FIFO up to [`UPLOAD_BUDGET_BYTES`] per frame and carrying the remainder
    /// over (the first attachment always stages, so an oversized one still
    /// makes progress). All readiness signals — the `Loaded` flip consumed by
    /// [`Self::get_best_tile`] and the [`TerrainTileReady`] message — are
    /// emitted here, on the frame the data reaches `uploading_tiles` and thus
    /// the GPU.
    pub(crate) fn stage_uploads(mut tile_atlases: Query<&mut TileAtlas>) {
        for mut tile_atlas in &mut tile_atlases {
            let mut staged_bytes = 0;
            while staged_bytes < UPLOAD_BUDGET_BYTES {
                let Some((tile, data)) = tile_atlas.pending_uploads.pop_front() else {
                    break;
                };
                staged_bytes += data.bytes().len();
                tile_atlas.stage_upload(tile, data);
            }
        }
    }

    fn stage_upload(&mut self, tile: AttachmentTile, data: AttachmentData) {
        // Backstop for the checks in `tile_loaded` and `request_tile`: an
        // entry queued for an evicted predecessor of this state must neither
        // decrement its `Loading` counter nor reach its slot.
        let Some(tile_state) = self
            .tile_states
            .get_mut(&tile.coordinate)
            .filter(|tile_state| tile_state.generation == tile.generation)
        else {
            return;
        };

        // The last outstanding attachment flips the tile to `Loaded`; that is the
        // once-per-tile edge at which the tile becomes sampleable.
        let became_ready = matches!(tile_state.state, LoadingState::Loading(1));

        tile_state.state = match tile_state.state {
            LoadingState::Loading(1) => LoadingState::Loaded,
            LoadingState::Loading(n) => LoadingState::Loading(n - 1),
            LoadingState::Loaded => {
                panic!("Loaded more attachments, than registered with the tile atlas.")
            }
        };

        self.uploading_tiles.push(AttachmentTileWithData {
            atlas_index: tile_state.atlas_index,
            label: tile.label,
            data,
        });

        if became_ready {
            self.ready_tiles
                .push((tile.coordinate, tile_state.atlas_index));
        }
    }

    /// Drains the pending residency changes into [`TerrainTileReady`] / [`TerrainTileDropped`]
    /// messages. Runs after [`Self::update`] so both edges of the frame are observed.
    pub(crate) fn emit_tile_events(
        mut tile_atlases: Query<(Entity, &mut TileAtlas)>,
        mut ready_messages: MessageWriter<TerrainTileReady>,
        mut dropped_messages: MessageWriter<TerrainTileDropped>,
    ) {
        for (terrain, mut tile_atlas) in &mut tile_atlases {
            let TileAtlas {
                ready_tiles,
                dropped_tiles,
                ..
            } = &mut *tile_atlas;

            ready_messages.write_batch(ready_tiles.drain(..).map(|(coordinate, atlas_index)| {
                TerrainTileReady {
                    terrain,
                    coordinate,
                    atlas_index,
                }
            }));

            dropped_messages.write_batch(dropped_tiles.drain(..).map(
                |(coordinate, atlas_index)| TerrainTileDropped {
                    terrain,
                    coordinate,
                    atlas_index,
                },
            ));
        }
    }

    /// Updates the tile atlas according to all corresponding tile_trees.
    pub(crate) fn update(
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        mut tile_atlases: Query<&mut TileAtlas>,
    ) {
        for (&(terrain, _view), tile_tree) in tile_trees.iter_mut() {
            // The terrain may have been despawned this frame, before
            // `cleanup_despawned_terrains` pruned its tile tree.
            let Ok(mut tile_atlas) = tile_atlases.get_mut(terrain) else {
                continue;
            };

            for tile_coordinate in tile_tree.released_tiles.drain(..) {
                tile_atlas.release_tile(tile_coordinate);
            }

            for tile_coordinate in tile_tree.requested_tiles.drain(..) {
                tile_atlas.request_tile(tile_coordinate);
            }
        }
    }

    fn request_tile(&mut self, tile_coordinate: TileCoordinate) {
        if !self.existing_tiles.contains(&tile_coordinate) {
            return;
        }

        // check if the tile is already present else start loading it
        if let Some(tile) = self.tile_states.get_mut(&tile_coordinate) {
            if tile.requests == 0 {
                // the tile is now used again
                self.unused_indices
                    .retain(|&atlas_index| tile.atlas_index != atlas_index);
            }

            tile.requests += 1;
        } else {
            let atlas_index = self
                .unused_indices
                .pop_front()
                .expect("Atlas out of indices");

            // Remove the tile still cached in this slot. If it was ready, report it
            // dropped in the same pass so its data is no longer considered available.
            let Self {
                tile_states,
                dropped_tiles,
                pending_uploads,
                ..
            } = self;
            tile_states.retain(|&coordinate, tile| {
                if tile.atlas_index != atlas_index {
                    return true;
                }
                if matches!(tile.state, LoadingState::Loaded) {
                    dropped_tiles.push((coordinate, atlas_index));
                }
                // Uploads still queued for the evicted tile must not reach the
                // slot's next occupant.
                pending_uploads.retain(|(pending, _)| pending.coordinate != coordinate);
                false
            });

            self.generation = self.generation.wrapping_add(1);

            self.tile_states.insert(
                tile_coordinate,
                TileState {
                    requests: 1,
                    state: LoadingState::Loading(self.attachments.len() as u32),
                    atlas_index,
                    generation: self.generation,
                },
            );

            for label in self.attachments.keys() {
                self.to_load.push(AttachmentTile {
                    coordinate: tile_coordinate,
                    label: label.clone(),
                    generation: self.generation,
                });
            }
        }
    }

    fn release_tile(&mut self, tile_coordinate: TileCoordinate) {
        if !self.existing_tiles.contains(&tile_coordinate) {
            return;
        }

        let tile = self.tile_states.get_mut(&tile_coordinate).unwrap();
        tile.requests -= 1;

        if tile.requests == 0 {
            self.unused_indices.push_back(tile.atlas_index);

            // Todo: we should cancel loading tiles, that have not yet started loading and a no longer requested
        }
    }
}
