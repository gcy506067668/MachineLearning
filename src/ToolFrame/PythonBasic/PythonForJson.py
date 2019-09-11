import json

#  map  to  json object string
def encodeMapToJson(map_data):
    json_obj = json.dumps(map_data)
    print(type(json_obj))
    pass


#string json object to map
def decodeJsonToMap(json_obj):
    map_data = json.load(json_obj)
    print(map_data)
    pass

map = {"name":"zhansan","age":13}
encodeMapToJson(map)