# File used to create directories instead of having to create them all by hand

import os

local_dir = os.path.dirname(__file__)

enemies = [1,2,3,4,5,6,7,8]
runs = [1,2,3,4,5,6,6,7,8,9,10]
for enemy in enemies:
    name = 'enemy-' +str (enemy)
    newdir = os.path.join(local_dir, name)
    try: 
        os.mkdir(newdir)
    except FileExistsError as e:
        print(e)
        pass
    print(newdir)
    for run in runs:
        name = 'run-' + str(run)
        newnewdir = os.path.join(newdir, name)
        try: 
            os.mkdir(newnewdir)
        except FileExistsError as e:
            print(e)
            pass
        print(newnewdir)



