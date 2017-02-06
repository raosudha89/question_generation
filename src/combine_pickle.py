import sys
import cPickle as p

if __name__ == "__main__":
	ubuntu = p.load(open(sys.argv[1], 'rb'))
	unix = p.load(open(sys.argv[2], 'rb'))
	superuser = p.load(open(sys.argv[3], 'rb'))
	combined = unix + superuser + ubuntu
	p.dump(combined, open(sys.argv[4], 'wb'))
