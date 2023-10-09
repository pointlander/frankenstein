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

	de, en := []byte(German[0]), []byte(English[0])
	var buffer [4]byte
	input := NewMatrix(0, 1024, len(de)+1+len(en)+2)
	train := make([]byte, len(de))
	copy(train, de)
	train = append(train, 0)
	train = append(train, en...)
	train = append(train, 0, 0)
	for i, symbol := range train {
		buffer[i%len(buffer)] = symbol
		for _, s := range buffer {
			embedding := make([]float64, 256)
			embedding[s] = 1
			input.Data = append(input.Data, embedding...)
		}
	}
	embedding, _ := PCA(input)
	embedding = Normalize(embedding)
	sa := SelfAttention(embedding, embedding, embedding)
	for i := 0; i < sa.Rows; i++ {
		fmt.Println(i, sa.Data[i*sa.Cols:(i+1)*sa.Cols])
	}
}
