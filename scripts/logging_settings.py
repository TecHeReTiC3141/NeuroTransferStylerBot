import logging

logging.basicConfig(filename=r'../log_set.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s: %(message)s',
                    datefmt='%H:%M:%S')
