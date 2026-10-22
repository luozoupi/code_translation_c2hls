class AXI:
  def __init__(self, name, data_width, addr_width):
    self.name = name
    self.data_width = data_width
    self.addr_width = addr_width

MAX_AXI_BRAM_ADDR_WIDTH = 22  # PATCH 1: was 20 (1 MB/port); 22 = 4 MB/port
