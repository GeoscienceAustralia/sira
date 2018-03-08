from modulefinder import ModuleFinder

finder = ModuleFinder()
finder.run_script('C:\\Users\\u12089\\Desktop\\sifra\\scripts\\find_imported_modules.py')

print('Loaded modules:')
for name, mod in finder.modules.items():
    print(','.join(list(mod.globalnames.keys())[:3]))

print('-'*50)
print('Modules not imported:')
print('\n'.join(finder.badmodules.keys()))