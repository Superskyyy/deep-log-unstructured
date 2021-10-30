import re

regex = '(?P<Token0>.*?)\s+(?P<Timestamp>.*?)\s+(?P<Date>.*?)\s+(?P<User>.*?)\s+(?P<Month>.*?)\s+(?P<Day>.*?)\s+(?P<Time>.+?)\s+(?P<Location>.*?)\s+(?P<Component>.*?)(\[(?P<PID>.*?)\])?:\s+(?P<Message>.*?)'

regex = re.compile('^' + regex + '$')

s = '- 1131529689 2005.11.09 tbird-admin1 Nov 10 01:48:09 local@tbird-admin1 postfix/postdrop[13930]: warning: unable to look up public/pickup: No such file or directory'
print(regex.search(s))

a = regex.search(s)

print(a.group('Message'))
