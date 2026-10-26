async def comfy_entrypoint():
    from .nodes import MossAFMExtension
    return MossAFMExtension()
