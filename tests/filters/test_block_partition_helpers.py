import unittest

from pyrecest.filters import contiguous_partition, resolve_partition, validate_partition


class BlockPartitionHelpersTest(unittest.TestCase):
    def test_contiguous_partition_covers_components(self):
        self.assertEqual(
            contiguous_partition(5, block_size=2),
            ((0, 1), (2, 3), (4,)),
        )

    def test_resolve_named_partitions(self):
        self.assertEqual(resolve_partition(3, None), ((0, 1, 2),))
        self.assertEqual(resolve_partition(3, "singleton"), ((0,), (1,), (2,)))
        self.assertEqual(resolve_partition(5, "contiguous"), ((0, 1, 2, 3), (4,)))

    def test_validate_partition_rejects_overlap_and_gaps(self):
        self.assertEqual(validate_partition([(0, 2), (1,)], 3), ((0, 2), (1,)))
        with self.assertRaises(ValueError):
            validate_partition([(0, 1), (1, 2)], 3)
        with self.assertRaises(ValueError):
            validate_partition([(0,)], 3)

    def test_partition_counts_must_be_integral(self):
        invalid_counts = (True, 3.5, "3", [3])

        for invalid_count in invalid_counts:
            with self.subTest(function="contiguous", count=invalid_count):
                with self.assertRaises(ValueError):
                    contiguous_partition(invalid_count, block_size=2)
            with self.subTest(function="resolve", count=invalid_count):
                with self.assertRaises(ValueError):
                    resolve_partition(invalid_count, "singleton")
            with self.subTest(function="validate", count=invalid_count):
                with self.assertRaises(ValueError):
                    validate_partition([(0,)], invalid_count)

    def test_contiguous_block_size_must_be_integral(self):
        for invalid_size in (True, 0, 2.5, "2", [2]):
            with self.subTest(block_size=invalid_size), self.assertRaises(ValueError):
                contiguous_partition(3, block_size=invalid_size)

    def test_partition_component_indices_must_be_integral(self):
        invalid_partitions = (
            [(0.5, 1), (2,)],
            [(True, 1), (2,)],
            [("0", 1), (2,)],
            [([0], 1), (2,)],
        )

        for invalid_partition in invalid_partitions:
            with self.subTest(partition=invalid_partition), self.assertRaises(
                ValueError
            ):
                validate_partition(invalid_partition, 3)


if __name__ == "__main__":
    unittest.main()
