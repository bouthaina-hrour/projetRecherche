import bpy
import csv

List = []

ob = bpy.data.objects
for obj in ob:
    print (obj)
    List.append(obj.name)

with open(r'./Results/GoodShoot.csv', 'w', newline='') as f:
    filenames = List
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(List)

    for frame in range(180):
        bpy.context.scene.frame_set(frame)
        list_pos = []
        for obj in ob:
            joint_name = obj.name.split('.')[0]  # Assuming the joint names are formatted as "joint_name.001"
            joint_pos = bpy.data.objects[joint_name].location  # Get the joint position in the current frame
            obj.location = joint_pos  # Move the object to the joint position
            list_pos.append("("+ str(joint_pos.x) + ", "+ str(joint_pos.z) + ")")
            print(obj)
            print(joint_pos)
            print("*"*30)
        print(list_pos)
        writer.writerow(list_pos)
