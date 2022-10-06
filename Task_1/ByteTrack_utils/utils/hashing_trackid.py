from zlib import crc32

def bytes_to_int(b):
    return int(crc32(b) & 0xffffffff) #/ 2**32

def str_to_int(s, encoding="utf-8"):
    return bytes_to_int(s.encode(encoding))