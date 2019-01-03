import logging

from cli import commands
from loader.MongoGateway import gateway

if __name__ == '__main__':
    logger = logging.getLogger("tgml")
    logger.setLevel(logging.DEBUG)
    try:
        commands()
    finally:
        gateway.close_connection()
