import json,os


#  notebook文件路径
notebook_file_path = "F:/project/assignment2"





def ipynbTOPython(path,output_path):
    files = os.listdir(path)

    for index, file in enumerate(files):

        if file.__contains__(".ipynb"):
            if os.path.isfile(os.path.join(path, file)):
                with open(os.path.join(path, file), "r") as f:
                    ipynb_data = json.load(f)
                    with open(os.path.join(output_path, file.replace(".ipynb",".py")),"w") as pyfile:
                        list_code = ipynb_data["cells"]

                        for i, cellsobj in enumerate(list_code):
                            if(cellsobj['cell_type'] == "code"):
                                pyfile.write("\n")
                                for j, code_line in enumerate(cellsobj["source"]):

                                    pyfile.write(code_line)
                                    pass
                            else:
                                pyfile.write("\n\'\'\'\n")
                                for j, code_line in enumerate(cellsobj["source"]):

                                    pyfile.write(code_line)
                                    pass
                                pyfile.write("\n\'\'\'")


            else:
                ipynbTOPython(os.path.join(path, file))

    pass



ipynbTOPython(notebook_file_path,"F:/project/MachineLearning/src/CourseHomework/CS231N/assignment2")
