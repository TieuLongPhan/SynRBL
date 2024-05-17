import pytest

from synrbl.SynUtils.batching import DataLoader, Dataset


@pytest.mark.parametrize(
    "data,batch_size,exp_results",
    [
        ([1], 1, [[1]]),
        ([1, 2], 1, [[1], [2]]),
        ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
    ],
)
def test_data_loader(data, batch_size, exp_results):
    loader = DataLoader(iter(data), batch_size=batch_size)
    for data, exp_data in zip(loader, exp_results):
        assert len(exp_data) == len(data)
        for a, b in zip(data, exp_data):
            print(a, b)
            assert b == a
    with pytest.raises(StopIteration):
        next(loader)


def test_dataset_init_from_list():
    data = ["A", "B", "C"]
    dataset = Dataset(data)
    assert "A" == next(dataset)
    assert "B" == next(dataset)
    assert "C" == next(dataset)
