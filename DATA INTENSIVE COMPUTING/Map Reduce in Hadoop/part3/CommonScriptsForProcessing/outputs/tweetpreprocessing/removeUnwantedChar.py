import os
#'ccNCAAarticle2', 'nytimesNBAArticle', 'nytimesNCAAArticle',
for folder in ['ccNBAarticle','ccNCAAarticle','ccNCAAarticle2', 'nytimesNBAArticle', 'nytimesNCAAArticle','tweets']:
    Filenames = next(os.walk(folder))[2]
    for f in Filenames:
        print(f)
        # Read in the file
        with open(folder+'/'+f, 'r') as file :
            filedata = file.read()
        filedata = str(filedata)
        filedata = filedata.replace('#', '')
        filedata = filedata.replace('@', '')
        filedata = filedata.replace('"', '')
        filedata = filedata.replace('*', '')
        filedata = filedata.replace('“', '')
        filedata = filedata.replace('`', '')
        filedata = filedata.replace('”', '')
        # Replace the target string
        # Write the file out again

        with open(folder+'/'+f, 'w') as file:
          file.write(filedata)
