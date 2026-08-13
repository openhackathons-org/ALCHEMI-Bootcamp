from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pytest
import torch
from helpers import (
    figure_to_html,
    load_molecule_collection,
    load_molecule_manifest,
    molecule_source_path,
    plot_record_layout,
    tutorial_workspace,
)
from helpers import lesson as lesson_helpers
from nvalchemi.data import (
    AtomicData,
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    Batch,
    DataLoader,
    Dataset,
    InMemoryDataset,
    Reader,
)
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]


def test_pinned_molecule_collection_identity() -> None:
    atoms, frame = load_molecule_collection(ROOT)

    assert len(atoms) == len(frame) == 32
    assert sum(map(len, atoms)) == 322
    assert frame.loc[frame["label"] == "Ethyne", "atoms"].item() == 4
    assert frame.loc[frame["label"] == "Phenol", "atoms"].item() == 13
    assert frame.loc[frame["label"] == "2,3-dimethylbutane", "atoms"].item() == 20
    assert molecule_source_path(ROOT).is_file()


def test_manifest_loading_does_not_materialize_extxyz_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("manifest loading must not read extxyz frames")

    monkeypatch.setattr(lesson_helpers, "read", fail_if_called)
    frame = load_molecule_manifest(ROOT)

    assert len(frame) == 32
    assert int(frame["atoms"].sum()) == 322
    assert frame.loc[23, ["label", "formula"]].to_dict() == {
        "label": "Phenol",
        "formula": "C6H6O",
    }


def test_tutorial_workspace_is_live_and_writable() -> None:
    owner, path = tutorial_workspace()
    marker = path / "marker.txt"
    marker.write_text("ready", encoding="utf-8")

    assert marker.read_text(encoding="utf-8") == "ready"
    owner.cleanup()
    assert not path.exists()


def test_record_layout_plot_labels_counts_and_boundaries() -> None:
    counts = [2, 3, 4]
    pointers = [0, 2, 5, 9]

    figure = plot_record_layout(counts, pointers)
    count_axis, pointer_axis = figure.axes

    assert count_axis.get_xlabel() == "Record index"
    assert count_axis.get_ylabel() == "Atoms per record"
    assert [bar.get_height() for bar in count_axis.patches] == counts
    assert pointer_axis.get_xlabel() == "Record boundary"
    assert pointer_axis.get_ylabel() == "Cumulative atom rows"
    assert pointer_axis.lines[0].get_ydata().tolist() == pointers
    assert not plt.fignum_exists(figure.number)

    html = figure_to_html(figure, 'Record sizes & "boundaries"')
    assert "data:image/png;base64," in html.data
    assert 'alt="Record sizes &amp; &quot;boundaries&quot;"' in html.data
    plt.close(figure)


def _write_representative_records() -> tuple[
    TemporaryDirectory[str], Path, list[AtomicData]
]:
    molecules, _ = load_molecule_collection(ROOT)
    records = []
    for record_id, molecule in enumerate(molecules[:3]):
        molecule = molecule.copy()
        molecule.set_cell([8.0 + record_id, 9.0, 10.0])
        molecule.set_pbc([True, True, True])
        molecule.info["charge"] = record_id - 1
        data = AtomicData.from_atoms(molecule, dtype=torch.float32)
        data.add_system_property("record_id", torch.tensor([record_id]))
        records.append(data)

    source_batch = Batch.from_data_list(records, device="cpu")
    owner, path = tutorial_workspace()
    store = path / "records.zarr"
    AtomicDataZarrWriter(store).write(source_batch)
    return owner, store, records


def test_zarr_round_trip_preserves_scientific_payload_and_field_levels() -> None:
    owner, store, records = _write_representative_records()
    reader = AtomicDataZarrReader(store)
    raw, metadata = reader.read(1)
    source = records[1]

    assert metadata["index"] == 1
    assert metadata["physical_index"] == "1"
    assert metadata["source_file"].endswith("records.zarr")
    for field in ("atomic_numbers", "positions", "cell", "pbc", "charge"):
        assert torch.equal(raw[field], getattr(source, field))
        assert raw[field].device.type == "cpu"

    assert raw["atomic_numbers"].dtype == torch.int32
    assert raw["positions"].dtype == raw["cell"].dtype == torch.float32
    assert raw["pbc"].dtype == torch.bool
    assert raw["charge"].dtype == torch.float32
    assert reader.field_levels["atomic_numbers"] == "atom"
    assert reader.field_levels["positions"] == "atom"
    assert reader.field_levels["cell"] == "system"
    assert reader.field_levels["pbc"] == "system"
    assert reader.field_levels["charge"] == "system"
    assert reader.field_levels["record_id"] == "system"

    reader.close()
    owner.cleanup()


def test_dataset_streamed_and_resident_paths_preserve_payload() -> None:
    owner, store, records = _write_representative_records()

    reader = AtomicDataZarrReader(store)
    ordered = reader.read_many([2, 0])
    assert [int(data["record_id"].item()) for data, _ in ordered] == [2, 0]

    dataset = Dataset(reader, device="cpu", num_workers=1)
    sample, metadata = dataset[1]
    assert isinstance(sample, AtomicData)
    assert metadata["index"] == 1
    assert metadata["physical_index"] == "1"
    assert metadata["source_file"].endswith("records.zarr")
    assert sample.device.type == "cpu"
    for field in ("atomic_numbers", "positions", "cell", "pbc", "charge"):
        assert torch.equal(getattr(sample, field), getattr(records[1], field))

    streamed = next(
        iter(DataLoader(dataset, batch_size=3, prefetch_factor=0, use_streams=False))
    )
    resident = InMemoryDataset(
        reader=AtomicDataZarrReader(store),
        chunk_size=2,
        device="cpu",
    )
    cached = next(
        iter(DataLoader(resident, batch_size=3, prefetch_factor=0, use_streams=False))
    )

    assert streamed.record_id.tolist() == cached.record_id.tolist() == [0, 1, 2]
    assert resident.in_memory_batch.device.type == "cpu"
    for field in (
        "atomic_numbers",
        "positions",
        "cell",
        "pbc",
        "charge",
        "record_id",
        "batch_ptr",
    ):
        assert torch.equal(getattr(streamed, field), getattr(cached, field))
        assert getattr(streamed, field).device.type == "cpu"

    dataset.close()
    resident.close()
    owner.cleanup()


def test_dataset_rejects_malformed_atom_level_payload() -> None:
    owner, store, _records = _write_representative_records()
    reader = AtomicDataZarrReader(store)
    raw, _metadata = reader.read(0)

    class MissingPositionReader(Reader):
        def __len__(self) -> int:
            return 1

        def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
            if index != 0:
                raise IndexError(index)
            sample = {key: value.clone() for key, value in raw.items()}
            sample["positions"] = sample["positions"][:-1]
            return sample

    invalid_dataset = Dataset(MissingPositionReader(), device="cpu")
    with pytest.raises(ValidationError, match="expected 4, got 3"):
        invalid_dataset[0]

    reader.close()
    invalid_dataset.close()
    owner.cleanup()
