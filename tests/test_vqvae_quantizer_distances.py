import torch

from src.vqvae.model import VectorQuantizer


def test_vector_quantizer_chunked_distance_matches_cdist() -> None:
    quantizer = VectorQuantizer(num_embeddings=11, embedding_dim=7).eval()
    flat_inputs = torch.randn(13, 7)

    distances = quantizer._compute_distances_chunked(flat_inputs, chunk_size=4)
    expected = torch.cdist(flat_inputs, quantizer.codebook.weight, p=2.0).pow(2)

    assert distances.shape == expected.shape
    assert torch.allclose(distances, expected, atol=1e-5, rtol=1e-4)
