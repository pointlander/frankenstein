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
func Load() int {
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
		German = append(German, line)
	}

	if len(English) != len(German) {
		panic("English and German data size not equal")
	}
	return len(English)
}

func main() {
	size := Load()
	fmt.Println("size: ", size)
}
