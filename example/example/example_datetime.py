
import datetime

begin = datetime.datetime.now()

j = 0
for i in range(1000 * 1000):
    j = j + 1

end = datetime.datetime.now()
delta = end - begin

print(end)
print(delta * 1000)