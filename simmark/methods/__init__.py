from .expmin import ExpMinProcessor, expmin_detect
from .synthid import SynthIDProcessor, synthid_detect
from .nomark import NoMarkProcessor
from .watermax import WaterMaxProcessor, watermax_detect


logit_processors = {
    "expmin": ExpMinProcessor,
    "synthid": SynthIDProcessor,
    "nomark": NoMarkProcessor,
    "watermax": WaterMaxProcessor
}

detection_methods = {
    "expmin": expmin_detect,
    "synthid": synthid_detect,
    "watermax": watermax_detect
}