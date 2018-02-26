#!/usr/bin/env python

import sys
import os
import argparse
import telop_vr_pb2 as telop_vr_pb
import numpy as np
from ipdb import set_trace as debug

def parse_protobufs(args):
  # init empty session
  telop_vr_session = telop_vr_pb.TelopVRSession()
  # read from file
  with open("{}/{}".format(args.data_folder, args.data_name), 'rb') as f:
    telop_vr_session.ParseFromString(f.read())
  # analysis of data
  #print(telop_vr_session)
  for state in telop_vr_session.states:
    #print(state)
    #print(state.timestamp)
    for item in state.items:
      #print(item)
      # print(item.id)
      pass

  return telop_vr_session

def parse_args():
  examples="""
Examples:
  python analysis.py -D /path/to/data -d _SessionStateData.proto
"""
  # add args
  parser = argparse.ArgumentParser(epilog=examples, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-D', dest='data_folder', type=str, default=os.path.dirname(os.path.realpath(__file__))+'/data', help='Path to data folder')
  parser.add_argument('-d', dest='data_name', type=str, default='_SessionStateData.proto', help='Name of datafile')

  # parse args
  args = parser.parse_args()

  return args



if __name__ == '__main__':
  data = parse_protobufs(parse_args())








