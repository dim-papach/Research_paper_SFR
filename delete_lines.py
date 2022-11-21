with open("Karachentsev_06_11_2022.txt", "r") as fp, open("Karachentsev_updated.txt","w") as new_fp:
    for line in fp:
        if "<"  not in line:
            new_fp.write(line)
