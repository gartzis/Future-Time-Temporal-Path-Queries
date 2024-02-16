def file_reader(file_name): #read a file one line at a time
    while True:
        row =file_name.readline()
        if not row:
            break
        yield row