from __future__ import absolute_import

from hls4ml.writer.writers import Writer, register_writer, get_writer
from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.writer.oneapi_writer import OneApiWriter

register_writer('Vivado', VivadoWriter)
register_writer('oneAPI', OneApiWriter)
