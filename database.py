# import os
# def get_folder_names(root_folder):
#     chroma_instances = {}
#     for root, dirs, files in os.walk(root_folder):
#         if root == root_folder:
#             for dir_name in dirs:
#                 chroma_instances[dir_name] = dir_name
#     return chroma_instances

# root_folder = "ChromaDB"
# chroma_instances = get_folder_names(root_folder)
# print(chroma_instances)
chroma_instances = {}