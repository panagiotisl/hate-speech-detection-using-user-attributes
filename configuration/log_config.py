import logging

logging.basicConfig(filename='log/app.log', filemode='a',
                    format='%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

logger = logging.getLogger()
