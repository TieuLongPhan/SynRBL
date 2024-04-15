import unittest
from synrbl.SynVis import ReactionVisualizer
from PIL import Image


class TestReactionVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = ReactionVisualizer()

    def test_visualize_reaction(self):
        # Test visualization of a single chemical reaction
        reaction_str = "C1=CC=CC=C1>>CCO"
        image = self.visualizer.visualize_reaction(reaction_str)
        self.assertIsInstance(image, Image.Image)
        # image.show()


if __name__ == "__main__":
    unittest.main()
