from .reference import ReferenceFakeQuantBackend, fake_quant_activation as fake_quant_activation


def get_backend(name):
    if name == "reference":
        return ReferenceFakeQuantBackend()
    if name == "cutlass":
        from .cutlass import CutlassW4A8Backend

        return CutlassW4A8Backend()
    raise ValueError(f"Unsupported backend: {name}")
