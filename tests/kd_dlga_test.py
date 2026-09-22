import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from kd.model.kd_dlga import KD_DLGA


# The original DLGA class depends on an NN class. For testing purposes,
# we need a placeholder that mimics the original structure.
# We can define a minimal mock NN class right inside the test file.
class MockNN(torch.nn.Module):
    def __init__(self, Input_Dim, Num_Hidden_Layers, Neurons_Per_Layer, Output_Dim, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(Input_Dim, Neurons_Per_Layer))
        for _ in range(Num_Hidden_Layers):
            self.layers.append(torch.nn.Linear(Neurons_Per_Layer, Neurons_Per_Layer))
        self.layers.append(torch.nn.Linear(Neurons_Per_Layer, Output_Dim))

    def forward(self, x):
        for layer in self.layers:
            x = torch.sin(layer(x))  # Using sin as in the original
        return x


def test_initialization_succeeds():
    """
    Tests that the KD_DLGA class can be initialized correctly.
    This test should already be passing from your colleague's work.
    """
    operators = ["u", "u_x"]
    model = KD_DLGA(operators=operators, epi=0.01, input_dim=2)
    assert model.user_operators == operators
    assert model.epi == 0.01


def test_generate_meta_data_override(tmp_path):
    """
    Tests that the overridden generate_meta_data method correctly
    builds the Theta matrix based on the user_operators list.
    """
    # Define the candidate library used to construct the metadata matrix.
    custom_operators = ["u", "u_x", "u_xx"]
    model = KD_DLGA(operators=custom_operators, epi=0.01, input_dim=2, max_iter=100)

    # Replace the network with a lightweight fixture and save its state.
    model.Net = MockNN(Input_Dim=2, Num_Hidden_Layers=1, Neurons_Per_Layer=10, Output_Dim=1)
    model_save_dir = tmp_path / "model_save"
    model_save_dir.mkdir()
    model.best_epoch = 500
    dummy_model_path = model_save_dir / f"Net_{model.best_epoch}.pkl"
    torch.save(model.Net.state_dict(), dummy_model_path)

    # Create a small coordinate array for metadata generation.
    X_test = np.random.rand(10, 2)  # 10 data points, 2 features (e.g., x and t)

    # Run from the temporary directory because the method reads ``model_save``.
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        model.generate_meta_data(X_test)

        # Each configured operator must produce one column.
        assert model.Theta is not None
        assert model.Theta.shape[1] == len(custom_operators)

    finally:
        # Restore the caller's working directory even when the assertion fails.
        os.chdir(original_cwd)


def test_random_genome_respects_operator_bounds():
    """Ensure generated genomes never reference operators outside the configured library."""
    # Use two operators so the only valid indices are 0 and 1.
    custom_operators = ["u", "u_x"]
    model = KD_DLGA(operators=custom_operators, epi=0.01, input_dim=2)

    # Generate a genome through the inherited implementation.
    # random_genome delegates individual modules to random_module.
    genome = model.random_genome()

    # Every generated gene must reference the custom operator list.
    # An index of 2 or greater would reveal a hard-coded operator bound.
    assert genome, "Genome should not be empty"
    for module in genome:
        for gene_index in module:
            assert gene_index < len(custom_operators)


def test_mutation_respects_operator_bounds():
    """Ensure mutation keeps every gene within the configured operator library."""
    # Keep the operator list at length two for a strict bound check.
    custom_operators = ["u", "u_x"]
    model = KD_DLGA(operators=custom_operators, epi=0.01, input_dim=2)

    # Seed a valid population and force mutation.
    # A 100 percent mutation rate makes the assertion deterministic.
    model.mutate_rate = 1.0
    model.pop_size = 1
    model.Chrom = [[[0], [1]]]  # This chromosome represents the valid modules [[u], [u_x]].

    # Mutate the prepared population.
    model.mutation()

    # Mutated genes must remain within the configured operator range.
    mutated_genome = model.Chrom[0]
    assert mutated_genome, "Mutated genome should not be empty"
    for module in mutated_genome:
        for gene_index in module:
            assert gene_index < len(custom_operators)


def test_convert_chrom_to_eq_uses_custom_operators():
    """Ensure equation rendering uses the configured operator names."""
    # Use a nonstandard operator order to verify name mapping.
    custom_operators = ["u_xx", "u", "u_t"]
    model = KD_DLGA(operators=custom_operators, epi=0.01, input_dim=2)

    # Construct a chromosome and its coefficients explicitly.
    # The chromosome represents u_t = 2.5 * u_xx - 1.0 * u.
    best_chrom = [[0], [1]]  # Its modules map to ['u_xx', 'u'] in that order.
    best_coef = np.array([[2.5], [-1.0]])
    left_hand_side = "u_t"

    # Convert the chromosome to an equation string.
    equation_str = model.convert_chrom_to_eq(best_chrom, left_hand_side, best_coef)

    # The output must preserve the custom operator names.
    # This guards against the former fixed mapping to 'u' and 'ux'.
    assert "2.5*u_xx" in equation_str
    assert "-1.0*u" in equation_str
    assert "u_t=" in equation_str
