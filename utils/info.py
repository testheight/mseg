import logging,os

def log_output(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    stream_handler = logging.StreamHandler()  # 日志控制台输出
    
    handler = logging.FileHandler(os.path.join(save_path,'info.log'), mode='w', encoding='UTF-8')  # 日志文件输出
    handler.setLevel(logging.DEBUG)

    # 控制台输出格式
    stream_format = logging.Formatter("Time: %(asctime)s -- INFO: %(message)s")
    # 文件输出格式
    logging_format = logging.Formatter("Time: %(asctime)s -- INFO: %(message)s")
    
    handler.setFormatter(logging_format)  # 为改处理器handler 选择一个格式化器
    stream_handler.setFormatter(stream_format)
    
    logger.addHandler(handler)  # 为记录器添加 处理方式Handler
    logger.addHandler(stream_handler)
    return logger