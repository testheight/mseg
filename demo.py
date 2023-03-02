import logging

# 1.创建一个logger（日志记录器）对象；
my_logger = logging.Logger("first_logger")

# 2.定义handler（日志处理器），决定把日志发到哪里；
my_handler = logging.FileHandler("/home/lijiangtao/Documents/code_file/mseg/demo.log")

# 3.设置日志级别（level）和输出格式Formatters（日志格式器）；
my_handler.setLevel(logging.INFO)
my_format = logging.Formatter("时间:%(asctime)s 日志信息:%(message)s 行号:%(lineno)d")

# 把handler添加到对应的logger中去。
my_handler.setFormatter(my_format)
my_logger.addHandler(my_handler)


# 使用：
my_logger.info("我是日志组件")

