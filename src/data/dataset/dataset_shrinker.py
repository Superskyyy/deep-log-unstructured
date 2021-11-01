# with open('bgl2', 'r') as f:
#     with open('bgl_small', 'w') as fo:
#         counter = 0
#         for line in f:
#             fo.write(line)
#             if counter >= 5000000:
#                 break
#             counter += 1
#
# with open('spirit2', 'r') as f:
#     with open('spirit_small', 'w') as fo:
#         counter = 0
#         for line in f:
#             fo.write(line)
#             if counter >= 5000000:
#                 break
#             counter += 1
#
# with open('liberty2', 'r') as f:
#     with open('liberty_small', 'w') as fo:
#         counter = 0
#         for line in f:
#             fo.write(line)
#             if counter >= 5000000:
#                 break
#             counter += 1
#
# with open('Intrepid_RAS_0901_0908_scrubbed', 'r', encoding="latin-1") as f:
#     with open('Intrepid_RAS_0901_0908_scrubbed_small', 'w', encoding="latin-1") as fo:
#         counter = 0
#         for line in f:
#             fo.write(line)
#             if counter >= 5000000:
#                 break
#             counter += 1

with open('tbird2', 'r', encoding="latin-1") as f:
    with open('tbird2_medium_200m_40step', 'w', encoding="latin-1") as fo:
        counter = 0
        for line in f:
            if counter % 40 == 0:
                fo.write(line)
            if counter >= 200000000:
                break
            counter += 1

