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

	randomforest "github.com/malaschitz/randomForest"
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
	train := make([]byte, len(de))
	copy(train, de)
	train = append(train, 1)
	train = append(train, en...)
	input := NewMatrix(0, 1024, len(train))
	output := make([]int, len(train))
	for i, symbol := range train {
		buffer[i%len(buffer)] = symbol
		for _, s := range buffer {
			embedding := make([]float64, 256)
			embedding[s] = 1
			input.Data = append(input.Data, embedding...)
		}
		if next := i + 1; next < len(train) {
			output = append(output, int(train[next]))
		}
	}
	output = append(output, 0)
	embedding, _ := PCA(input)
	embedding = Normalize(embedding)
	sa := SelfAttention(embedding, embedding, embedding)
	in := make([][]float64, len(train))
	for i := 0; i < sa.Rows; i++ {
		in[i] = sa.Data[i*sa.Cols : (i+1)*sa.Cols]
	}

	forest := randomforest.Forest{}
	forest.Data = randomforest.ForestData{X: in, Class: output}
	fmt.Println("training...")
	forest.Train(1000)
}
