import torch
from packaging import version

def test_backend_version_parsing():
    """
    Test that version strings for available backends (CUDA, HIP) 
    are compliant with PEP 440 to ensure packaging.version.parse works.
    This test leverages torch.backends.cuda.is_built to check availability
    and applies the same parsing logic to torch.version.hip.
    """
    
    # Test CUDA version parsing if CUDA is built
    # Leveraging the similar API: torch.backends.cuda.is_built
    if torch.backends.cuda.is_built():
        cuda_version = torch.version.cuda
        try:
            parsed_version = version.parse(cuda_version)
            print(f"CUDA version parsed successfully: {parsed_version}")
        except version.InvalidVersion as e:
            print(f"Error parsing CUDA version '{cuda_version}': {e}")
            raise

    # Test HIP version parsing if HIP is available
    # Preserving the original bug reproduction logic for torch.version.hip
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        hip_version = torch.version.hip
        try:
            parsed_version = version.parse(hip_version)
            print(f"HIP version parsed successfully: {parsed_version}")
        except version.InvalidVersion as e:
            print(f"Error parsing HIP version '{hip_version}': {e}")
            raise

if __name__ == "__main__":
    test_backend_version_parsing()