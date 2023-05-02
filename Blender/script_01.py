import bpy


ob = bpy.context.scene.objects['Camera'] 

ob = bpy.context.object

for f in ob.animation_data.action.fcurves:
    for k in f.keyframe_points:
        fr = k.co[0]
        bpy.context.scene.frame_set(fr)
        pos=ob.location
        print(pos)
        print(ob)
        print("-"*20)
        

        