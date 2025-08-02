import re
import json


def strip_sub(s: str):
    return s.replace("<sub>", "").replace("</sub>", "")


lines: list[str] = []
with open("./vscode-icons.wiki/ListOfFiles.md", "r") as f:
    read = False
    for line in f.readlines():
        if line == "| :---: | :---: | :---: | :---: |\n":
            read = True
            continue
        if read:
            lines.append(line.strip())

regex = r'!\[([^\]]+)\]\(([^\)]+)\)'

layer1 = []


def category(s: str):
    if s.find(" ") > -1:
        return (None, None)
    if s.startswith("*"):
        return ("filename", s.strip("*"))
    if s.startswith("`"):
        return ("language-id", s.strip("`"))
    return ("extension", s)


for line in lines:
    data = [
        strip_sub(i).strip() for i in line.split("|")[1:-1]
    ]
    name = data[0]
    files = data[1]
    icon = data[2]

    match = re.search(regex, icon)

    if match:
        icon_file = match.group(2)
        layer1.append(([category(i.strip())
                      for i in files.split(",")], icon_file))

filenames = {}
language_ids = {}
extensions = {}

for data, url in layer1:
    for meta, text in data:
        url = url.replace(
            "https://github.com/vscode-icons/vscode-icons/blob/master/icons/", "")
        if meta == "filename":
            filenames[text] = url
        if meta == "extension":
            extensions[text] = url
        if meta == "language-id":
            language_ids[text] = url

with open("./icons-meta/filenames.ts", "w") as f:
    f.write("export const table = "+json.dumps(filenames, indent=4))

with open("./icons-meta/extensions.ts", "w") as f:
    f.write("export const table = "+json.dumps(extensions, indent=4))

with open("./icons-meta/language_ids.ts", "w") as f:
    f.write("export const table = "+json.dumps(language_ids, indent=4))
