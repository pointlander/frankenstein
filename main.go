// Copyright Frankenstein The Mach Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

var (
	English = make([]string, 0, 1024)
	German  = make([]string, 0, 1024)
)

// Load data from files
func Load() (length, max int) {
	in, err := os.Open("train.en")
	if err != nil {
		panic(err)
	}
	defer in.Close()

	reader := bufio.NewReader(in)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
		}
		line = strings.TrimSpace(line)
		if len(line) > max {
			max = len(line)
		}
		English = append(English, line)
	}

	in, err = os.Open("train.de")
	if err != nil {
		panic(err)
	}
	defer in.Close()

	reader = bufio.NewReader(in)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
		}
		line = strings.TrimSpace(line)
		if len(line) > max {
			max = len(line)
		}
		German = append(German, line)
	}

	length = len(English)
	if length != len(German) {
		panic("English and German data size not equal")
	}
	return length, max
}

func main() {
	size, max := Load()
	fmt.Println("size: ", size, " max: ", max)
}
