import colorsys
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import pretty_midi
from matplotlib.ticker import MaxNLocator


class SuffixTreeNode:
    def __init__(self):
        self.children = defaultdict(SuffixTreeNode)
        self.indices = []  # Starting indices in the original sequence


class SuffixTree:
    def __init__(self, sequence):
        self.root = SuffixTreeNode()
        self.sequence = sequence
        self._build(sequence)

    def _build(self, sequence):
        n = len(sequence)
        for i in range(n):
            self._insert_suffix(sequence[i:], i)

    def _insert_suffix(self, suffix, index):
        node = self.root
        node.indices.append(index)
        for element in suffix:
            node = node.children[element]
            node.indices.append(index)

    def find_repeated_patterns(self, min_length=2, min_occurrences=2):
        # Returns list of tuples: [(pattern_tuple, [start_indices]), ...], sorted.
        results = []
        for element, child_node in self.root.children.items():
            self._find_repeated_patterns_helper(
                child_node, [element], min_length, min_occurrences, results
            )

        results.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)
        return results

    def _find_repeated_patterns_helper(
        self, node, current_pattern, min_length, min_occurrences, results
    ):
        unique_indices = sorted(list(set(node.indices)))

        if (
            len(current_pattern) >= min_length
            and len(unique_indices) >= min_occurrences
        ):
            pattern_tuple = tuple(current_pattern)
            results.append((pattern_tuple, unique_indices))

        for element, child_node in node.children.items():
            self._find_repeated_patterns_helper(
                child_node,
                current_pattern + [element],
                min_length,
                min_occurrences,
                results,
            )


def get_channel(midi, channel_name=None):
    if channel_name:
        for instrument in midi.instruments:
            if instrument.name.strip().lower() == channel_name.strip().lower():
                if instrument.notes:
                    return instrument
                else:
                    print(
                        f"Warning: Found instrument '{instrument.name}' but it has no notes."
                    )

    for instrument in midi.instruments:
        if instrument.notes:
            return instrument

    if midi.instruments:
        print(
            "Warning: No instrument with notes found. Returning the first instrument."
        )
        return midi.instruments[0]

    raise ValueError("No instruments found in MIDI file")


def extract_interval_time_sequence(notes, time_tolerance):
    if not notes:
        return []

    # First element is dummy (0,0) as seq[i] links notes[i-1] and notes[i].
    sequence = [(0, 0)] * len(notes)

    # Use a small epsilon for floating point comparisons if tolerance is zero
    epsilon = sys.float_info.epsilon

    for i in range(1, len(notes)):
        interval = notes[i].pitch - notes[i - 1].pitch
        delta_time = notes[i].start - notes[i - 1].start

        if time_tolerance > epsilon:
            # Quantize: group deltas into buckets defined by tolerance
            quantized_delta = round(delta_time / time_tolerance)
        else:
            # No tolerance: use the actual float delta time.
            quantized_delta = delta_time

        sequence[i] = (interval, quantized_delta)

    return sequence


def analyze_midi_patterns(
    midi_path, channel_name=None, min_length=3, min_occurrences=2, time_tolerance=0.05
):
    try:
        midi_path_str = str(midi_path)
        midi = pretty_midi.PrettyMIDI(midi_path_str)
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return None, [], []

    try:
        channel = get_channel(midi, channel_name)
    except ValueError as e:
        print(f"Error getting channel in {midi_path}: {e}")
        return [], [], []

    # Sort notes by start time, then pitch for deterministic order
    notes = sorted(channel.notes, key=lambda note: (note.start, note.pitch))

    if not notes:
        print(f"No notes found in the selected channel for {midi_path}.")
        return [], [], []

    sequence = extract_interval_time_sequence(notes, time_tolerance)

    # SuffixTree min_length is number of sequence elements, which is notes - 1.
    min_sequence_len = min_length - 1
    if min_sequence_len < 1:
        min_sequence_len = 1

    # Need len(sequence) >= min_sequence_len + 1 because of the dummy element.
    if len(sequence) < min_sequence_len + 1:
        print(
            f"Track too short ({len(notes)} notes) for minimum pattern length ({min_length} notes)."
        )
        return notes, sequence, []

    print(f"Building Suffix Tree for sequence of length {len(sequence)}...")
    tree = SuffixTree(sequence)
    print("Finding repeated patterns...")

    patterns = tree.find_repeated_patterns(min_sequence_len, min_occurrences)
    print(f"Found {len(patterns)} raw patterns matching criteria.")

    return notes, sequence, patterns


# Holds info about one pattern occurrence. Indices refer to the NOTES list.
Occurrence = namedtuple(
    "Occurrence", ["start_idx", "end_idx", "weight", "pattern", "pattern_id"]
)


def get_distinct_colors(n):
    if n <= 0:
        return []
    colors = []
    saturation, value = 0.85, 0.85
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


def piano_roll_with_optimal_patterns(
    notes,
    optimal_occurrences,
    title="Piano Roll with Optimal Patterns",
    time_tolerance=0.0,
):
    if not notes:
        print("No notes to display.")
        return None, None

    fig, ax = plt.subplots(figsize=(18, 9))

    # Draw background notes
    for note in notes:
        rect = plt.Rectangle(
            (note.start, note.pitch - 0.5),
            note.end - note.start,
            1,
            facecolor="lightgray",
            edgecolor="darkgray",
            linewidth=0.5,
            zorder=1,
        )
        ax.add_patch(rect)

    legend_patches = {}

    if not optimal_occurrences:
        print("No optimal patterns found or provided to highlight.")
    else:
        unique_pattern_ids = sorted(
            list(set(occ.pattern_id for occ in optimal_occurrences))
        )
        colors = get_distinct_colors(len(unique_pattern_ids))
        pattern_id_to_color = {
            pid: colors[i] for i, pid in enumerate(unique_pattern_ids)
        }

        for occ in optimal_occurrences:
            color = pattern_id_to_color[occ.pattern_id]

            if (
                occ.start_idx >= len(notes)
                or occ.end_idx >= len(notes)
                or occ.start_idx > occ.end_idx
            ):
                print(f"Warning: Invalid indices in Occurrence: {occ}. Skipping.")
                continue

            pattern_notes = notes[occ.start_idx : occ.end_idx + 1]
            if not pattern_notes:
                continue

            for note_idx in range(occ.start_idx, occ.end_idx + 1):
                note = notes[note_idx]
                rect = plt.Rectangle(
                    (note.start, note.pitch - 0.5),
                    note.end - note.start,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.7,
                    zorder=5,
                )
                ax.add_patch(rect)

            min_start = min(n.start for n in pattern_notes)
            max_end = max(n.end for n in pattern_notes)
            min_pitch = min(n.pitch for n in pattern_notes)
            max_pitch = max(n.pitch for n in pattern_notes)
            box = plt.Rectangle(
                (min_start, min_pitch - 0.7),
                max_end - min_start,
                (max_pitch - min_pitch) + 1.4,
                facecolor="none",
                edgecolor=color,
                linewidth=2.0,
                linestyle="-",
                alpha=0.9,
                zorder=10,
            )
            ax.add_patch(box)

            if occ.pattern_id not in legend_patches:
                pattern_str_elements = []
                for interval, q_delta in occ.pattern:
                    pattern_str_elements.append(f"{interval}")  # Legend: Intervals only

                pattern_str = ", ".join(pattern_str_elements)
                # Weight in Occurrence is L_notes (length in notes)
                L_notes_str = f"L={occ.weight}"
                if len(pattern_str) > 40:
                    pattern_str = pattern_str[:37] + "..."
                label = f"{pattern_str} ({L_notes_str})"
                legend_patches[occ.pattern_id] = plt.Rectangle(
                    (0, 0), 1, 1, color=color, alpha=0.7, label=label
                )

    min_pitch_val = min(n.pitch for n in notes) if notes else 20
    max_pitch_val = max(n.pitch for n in notes) if notes else 100
    last_note_end = max(n.end for n in notes) if notes else 10

    ax.set_ylim(min_pitch_val - 5, max_pitch_val + 5)
    ax.set_xlim(0, last_note_end + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)

    if legend_patches:
        sorted_patches = [legend_patches[pid] for pid in sorted(legend_patches.keys())]
        ax.legend(
            handles=sorted_patches,
            loc="upper right",
            bbox_to_anchor=(1, 1),
            fontsize="small",
            title="Patterns",
        )

    plt.tight_layout()
    return fig, ax


def calculate_optimal_compression(notes, sequence, patterns, min_occurrences_final=2):
    print("--- Starting Optimal Compression Calculation ---")

    if not notes or not patterns:
        print("No notes or patterns to process.")
        return []

    all_occurrences_by_pattern = defaultdict(list)
    pattern_map = {}
    pattern_counter = 0
    total_potential_occurrences = 0

    for pattern_tuple, sequence_indices in patterns:
        L_pattern = len(pattern_tuple)
        L_notes = L_pattern + 1

        if len(sequence_indices) < min_occurrences_final:
            continue

        if pattern_tuple not in pattern_map:
            pattern_map[pattern_tuple] = pattern_counter
            pid = pattern_counter
            pattern_counter += 1
        else:
            pid = pattern_map[pattern_tuple]

        # --- Convert sequence indices to note indices ---
        # sequence[i] relates note[i-1] and note[i].
        # A pattern P of length L_pattern starting at sequence index `seq_idx`
        # corresponds to the notes from index `seq_idx - 1` to `seq_idx + L_pattern - 1`.
        # Start Note Index = seq_idx - 1
        # End Note Index = seq_idx + L_pattern - 1 (inclusive)
        # This mapping is crucial for correctly identifying the notes covered by a pattern occurrence.
        for seq_idx in sequence_indices:
            if seq_idx == 0:
                continue

            start_note_idx = seq_idx - 1
            end_note_idx = seq_idx + L_pattern - 1

            if (
                start_note_idx < 0
                or end_note_idx >= len(notes)
                or start_note_idx > end_note_idx
            ):
                print(
                    f"Warning: Skipping occurrence: Invalid note indices {start_note_idx}-{end_note_idx} from seq_idx {seq_idx}, L_pattern {L_pattern}. Max note idx: {len(notes) - 1}"
                )
                continue

            weight = L_notes

            occ = Occurrence(
                start_idx=start_note_idx,
                end_idx=end_note_idx,
                weight=weight,
                pattern=pattern_tuple,
                pattern_id=pid,
            )
            all_occurrences_by_pattern[pid].append(occ)
            total_potential_occurrences += 1

    print(
        f"Generated {total_potential_occurrences} potential Occurrence objects for {len(all_occurrences_by_pattern)} pattern types."
    )

    # Score pattern types based on potential saving heuristic: (L_notes * k) - (L_notes + k)
    pattern_savings = []
    for pid, occurrences in all_occurrences_by_pattern.items():
        if not occurrences:
            continue
        L_notes = occurrences[0].weight
        k_total = len(occurrences)

        if k_total < min_occurrences_final:
            saving = -float("inf")  # Mark ineligible
        else:
            saving = (L_notes * k_total) - (L_notes + k_total)

        pattern_savings.append(
            {
                "pid": pid,
                "saving": saving,
                "occurrences": sorted(occurrences, key=lambda x: x.start_idx),
                "L_notes": L_notes,
            }
        )

    pattern_savings.sort(key=lambda x: x["saving"], reverse=True)

    # Greedy Selection
    selected_occurrences_greedy = []
    covered_note_indices = set()

    print(
        f"Starting greedy selection from {len(pattern_savings)} eligible pattern types..."
    )
    patterns_considered = 0
    for pattern_info in pattern_savings:
        if math.isinf(pattern_info["saving"]) and pattern_info["saving"] < 0:
            continue
        patterns_considered += 1

        for occ in pattern_info["occurrences"]:
            occ_indices = set(range(occ.start_idx, occ.end_idx + 1))
            if not occ_indices.intersection(covered_note_indices):
                selected_occurrences_greedy.append(occ)
                covered_note_indices.update(occ_indices)

    print(f"Considered {patterns_considered} pattern types in greedy selection.")
    print(
        f"Greedy selection yielded {len(selected_occurrences_greedy)} occurrences before final filtering."
    )

    # Post-filter based on final counts
    if not selected_occurrences_greedy:
        print("No occurrences selected by greedy algorithm.")
        return []

    final_counts = Counter(occ.pattern_id for occ in selected_occurrences_greedy)
    final_filtered_occurrences = [
        occ
        for occ in selected_occurrences_greedy
        if final_counts[occ.pattern_id] >= min_occurrences_final
    ]

    print(
        f"Filtering to keep patterns used >= {min_occurrences_final} times in the selected set."
    )
    num_final_patterns = len(set(occ.pattern_id for occ in final_filtered_occurrences))
    print(
        f"Final optimal set: {len(final_filtered_occurrences)} occurrences from {num_final_patterns} pattern types."
    )

    final_filtered_occurrences.sort(key=lambda x: x.start_idx)
    return final_filtered_occurrences


def calculate_pattern_statistics(
    notes, optimal_occurrences, min_occurrences_final, time_tolerance
):
    stats = {
        "total_notes_in_track": len(notes),
        "total_patterns_found": 0,
        "total_pattern_occurrences": len(optimal_occurrences),
        "total_notes_covered_by_patterns": 0,
        "coverage_percentage": 0.0,
        "total_estimated_compression_saving": 0,
        "pattern_details": [],
        "time_tolerance": time_tolerance,
    }

    if not notes or not optimal_occurrences:
        if not notes:
            print("Warning: Cannot calculate stats - no notes.")
        if notes and not optimal_occurrences:
            print("Calculating stats: No optimal patterns found.")
        return stats

    occurrences_by_pid = defaultdict(list)
    for occ in optimal_occurrences:
        occurrences_by_pid[occ.pattern_id].append(occ)

    stats["total_patterns_found"] = len(occurrences_by_pid)
    total_saving, total_notes_covered = 0, 0
    epsilon = sys.float_info.epsilon

    for pid, occurrences in occurrences_by_pid.items():
        if not occurrences:
            continue

        occurrences.sort(key=lambda x: x.start_idx)
        pattern_tuple = occurrences[0].pattern
        L_notes = occurrences[0].weight
        k = len(occurrences)

        if k < min_occurrences_final:
            continue

        pattern_saving = (L_notes * k) - (L_notes + k)
        total_saving += pattern_saving
        total_notes_covered += L_notes * k
        start_indices = sorted([occ.start_idx for occ in occurrences])

        # --- Analyze Repetitions by Pitch Level ---
        # Group occurrences based on the absolute pitches of their notes, using the first valid occurrence as reference.
        internal_pitch_level_counts = Counter()
        ref_pitches = None
        for ref_occ in occurrences:
            start, end = ref_occ.start_idx, ref_occ.end_idx
            if 0 <= start < len(notes) and start <= end < len(notes):
                ref_pitches_candidate = tuple(
                    notes[i].pitch for i in range(start, end + 1)
                )
                if ref_pitches_candidate:
                    ref_pitches = ref_pitches_candidate
                    break

        if ref_pitches is None:
            print(
                f"Warning: Could not establish pitch reference for P{pid}. Grouping by pitch level failed."
            )
            if k > 0:
                internal_pitch_level_counts["unclassified"] = k
        else:
            # Compare all occurrences to the reference pitch sequence.
            for current_occ in occurrences:
                start, end = current_occ.start_idx, current_occ.end_idx
                if 0 <= start < len(notes) and start <= end < len(notes):
                    current_pitches = tuple(
                        notes[i].pitch for i in range(start, end + 1)
                    )
                    if len(current_pitches) == len(ref_pitches):
                        # Difference in first pitch determines transposition level for grouping.
                        transposition_level = current_pitches[0] - ref_pitches[0]
                        internal_pitch_level_counts[transposition_level] += 1
                    else:
                        print(
                            f"Warning: Pitch sequence length mismatch for P{pid} at index {start}. Marking as 'error'."
                        )
                        internal_pitch_level_counts["error"] += 1
                else:
                    print(
                        f"Warning: Invalid note indices [{start}-{end}] for P{pid} occurrence. Marking as 'invalid_index'."
                    )
                    internal_pitch_level_counts["invalid_index"] += 1

        # Extract the counts of occurrences at each found pitch level.
        repetition_counts = sorted(internal_pitch_level_counts.values(), reverse=True)

        pattern_str_elements = []
        for interval, q_delta in pattern_tuple:
            if time_tolerance > epsilon:
                approx_delta = q_delta * time_tolerance
                delta_repr = f"~{approx_delta:.2f}s"
            else:
                delta_repr = f"{q_delta:.3f}s"
            pattern_str_elements.append(f"({interval},{delta_repr})")
        pattern_display_str = ", ".join(pattern_str_elements)

        pattern_stat = {
            "pattern_id": pid,
            "pattern_tuple": pattern_tuple,
            "pattern_display_str": pattern_display_str,
            "length_notes": L_notes,
            "occurrences": k,
            "repetition_counts_by_pitch_level": repetition_counts,
            "start_note_indices": start_indices,
            "estimated_saving": pattern_saving,
        }
        stats["pattern_details"].append(pattern_stat)

    stats["total_notes_covered_by_patterns"] = total_notes_covered
    stats["total_estimated_compression_saving"] = total_saving
    if stats["total_notes_in_track"] > 0:
        stats["coverage_percentage"] = (
            total_notes_covered / stats["total_notes_in_track"]
        ) * 100
    else:
        stats["coverage_percentage"] = 0.0

    stats["pattern_details"].sort(
        key=lambda x: (-x["estimated_saving"], -x["length_notes"])
    )

    return stats


# Full pipeline
def analyze_and_visualize_optimal_patterns(
    midi_path,
    channel_name=None,
    min_length=3,
    min_occurrences=2,
    time_tolerance=0.05,
    save_path=None,
):
    midi_path = Path(midi_path)
    print(f"\n--- Analyzing {midi_path.name} ---")
    print(
        f"Parameters: min_length(notes)={min_length}, min_final_occurrences={min_occurrences}, time_tolerance={time_tolerance}s"
    )
    if channel_name:
        print(f"Target Channel: {channel_name}")

    notes, sequence, raw_patterns = analyze_midi_patterns(
        midi_path,
        channel_name,
        min_length,
        min_occurrences,
        time_tolerance,
    )

    stats_args = {
        "notes": notes if notes is not None else [],
        "optimal_occurrences": [],
        "min_occurrences_final": min_occurrences,
        "time_tolerance": time_tolerance,
    }

    if notes is None:
        print("MIDI loading failed. Cannot proceed.")
        stats = calculate_pattern_statistics(**stats_args)
        return None, None, stats
    if not notes:
        print("No notes found in the selected channel.")
        stats = calculate_pattern_statistics(**stats_args)
        return None, None, stats
    if not raw_patterns:
        print(
            f"No repeated patterns found meeting initial criteria (L>={min_length}, k>={min_occurrences})."
        )
        stats = calculate_pattern_statistics(**stats_args)
        return None, None, stats

    print(
        f"Found {len(raw_patterns)} distinct raw pattern types meeting initial criteria."
    )

    print(f"\nCalculating optimal pattern coverage (final k >= {min_occurrences})...")
    optimal_occurrences = calculate_optimal_compression(
        notes,
        sequence,
        raw_patterns,
        min_occurrences,
    )
    optimization_method = "Compression"

    stats_args["optimal_occurrences"] = optimal_occurrences

    print("\n--- Calculating Final Statistics ---")
    statistics = calculate_pattern_statistics(**stats_args)

    print("\n--- Analysis Statistics ---")
    print(f"Track: {midi_path.name}")
    print(f"Time Tolerance: {time_tolerance}s")
    print(f"Total Notes Analyzed: {statistics['total_notes_in_track']}")
    print(
        f"Final Unique Patterns (k>={min_occurrences}): {statistics['total_patterns_found']}"
    )
    print(
        f"Total Occurrences of Final Patterns: {statistics['total_pattern_occurrences']}"
    )
    print(
        f"Notes Covered by Final Patterns: {statistics['total_notes_covered_by_patterns']} ({statistics['coverage_percentage']:.2f}%)"
    )
    print(
        f"Total Estimated Compression Saving: {statistics['total_estimated_compression_saving']}"
    )
    print("---------------------------")

    if statistics["pattern_details"]:
        print("Details of Final Patterns (Sorted by Saving Score):")
        for i, p_stat in enumerate(statistics["pattern_details"]):
            pattern_str = p_stat["pattern_display_str"]
            if len(pattern_str) > 70:
                pattern_str = pattern_str[:67] + "..."

            print(f"  Pattern {p_stat['pattern_id']} (Rank {i + 1}):")
            print(f"    Pattern (Int, Δt): {pattern_str}")
            print(f"    Length: {p_stat['length_notes']} notes")
            print(f"    Occurrences (k): {p_stat['occurrences']}")

            counts_list = p_stat.get("repetition_counts_by_pitch_level", [])
            if counts_list:
                counts_str = ", ".join(map(str, counts_list))
                print(f"    Counts per Pitch Level: [{counts_str}]")
            else:
                print("    Counts per Pitch Level: (N/A or ungrouped)")

            print(f"    Est. Saving: {p_stat['estimated_saving']}")
            print("-" * 10)
    else:
        print(
            f"No patterns met the final criteria (k>={min_occurrences}) after optimization."
        )
    print("--- End Statistics ---")

    fig, ax = None, None
    if not optimal_occurrences and not notes:
        print("Skipping visualization: No notes available.")
    elif not optimal_occurrences:
        print(
            "Visualizing piano roll without optimal patterns (none met final criteria)."
        )
        title = f"{midi_path.name} (No Optimal Patterns, k>={min_occurrences}, tol={time_tolerance:.2f}s)"
        if channel_name:
            title = f"{midi_path.name}, {channel_name} (...)"
        fig, ax = piano_roll_with_optimal_patterns(
            notes, [], title=title, time_tolerance=time_tolerance
        )
    else:
        print("Visualizing piano roll with optimal patterns...")
        title = f"{midi_path.name}"
        if channel_name:
            title += f", {channel_name}"
        title += f" (k≥{min_occurrences}, tol={time_tolerance:.2f}s)"
        fig, ax = piano_roll_with_optimal_patterns(
            notes, optimal_occurrences, title=title, time_tolerance=time_tolerance
        )

    if fig:
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tol_str = f"{time_tolerance:.2f}".replace(".", "p")
            save_name = f"{midi_path.stem}_{optimization_method}_k{min_occurrences}_L{min_length}_tol{tol_str}{save_path.suffix}"
            full_save_path = save_path.parent / save_name
            try:
                fig.savefig(full_save_path, dpi=300, bbox_inches="tight")
                print(f"Visualization saved to {full_save_path}")
            except Exception as e:
                print(f"Error saving plot to {full_save_path}: {e}")
            plt.close(fig)
        else:
            plt.show()

    return fig, ax, statistics
