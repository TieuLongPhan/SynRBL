import sys
from pathlib import Path
import unittest
from rdkit import Chem
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.SynVis import ReactionVisualizer  
from PIL import Image
import matplotlib.pyplot as plt

class TestReactionVisualizer(unittest.TestCase):

    def setUp(self):
        self.visualizer = ReactionVisualizer()

    def test_visualize_reaction(self):
        # Test visualization of a single chemical reaction
        reaction_str = 'C1=CC=CC=C1>>CCO'
        image = self.visualizer.visualize_reaction(reaction_str)
        self.assertIsInstance(image, Image.Image)
        image.show()

    def test_plot_reactions(self):
        # Test plotting of reactions
        old_reaction_str = 'C1=CC=CC=C1>>CCO'
        new_reaction_str = 'C1=CC=CC=C1>>CCO.CN'
        self.visualizer.plot_reactions(old_reaction_str, new_reaction_str, savefig=False, pathname = root_dir / 'Test/SynVis/Test.png')

        # Check if matplotlib figure is created
        self.assertTrue(plt.fignum_exists(1))

if __name__ == '__main__':
    unittest.main()
