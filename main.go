// Copyright Frankenstein The Mach Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	randomforest "github.com/malaschitz/randomForest"
)

var (
	English = make([]string, 0, 256)
	German  = make([]string, 0, 256)
)

// Load data from files
func Load(n int) (length, maxen, maxde int) {
	in, err := os.Open("train.en")
	if err != nil {
		panic(err)
	}
	defer in.Close()

	reader, index := bufio.NewReader(in), 0
	for index < n {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
		}
		line = strings.TrimSpace(line)
		if len(line) > maxen {
			maxen = len(line)
		}
		English = append(English, line)
		index++
	}

	in, err = os.Open("train.de")
	if err != nil {
		panic(err)
	}
	defer in.Close()

	reader, index = bufio.NewReader(in), 0
	for index < n {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
		}
		line = strings.TrimSpace(line)
		if len(line) > maxde {
			maxde = len(line)
		}
		German = append(German, line)
		index++
	}

	length = len(English)
	if length != len(German) {
		panic("English and German data size not equal")
	}
	return length, maxen, maxde
}

var (
	// FlagCount is the number of lines of training data to load
	FlagCount = flag.Int("count", 256, "number of lines of training to load")
)

func main() {
	flag.Parse()

	size, maxen, maxde := Load(*FlagCount)
	max := maxen + maxde + 1
	fmt.Println("size: ", size, " max: ", max)

	in := make([][]float64, 0, *FlagCount)
	output := make([]int, 0, *FlagCount)
	for pair := 0; pair < len(German); pair++ {
		fmt.Println("loading", pair)
		de, en := []byte(German[pair]), []byte(English[pair])
		var buffer [4]byte
		train := make([]byte, len(de))
		copy(train, de)
		train = append(train, 1)
		train = append(train, en...)
		input := NewMatrix(0, 1024, max)

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
		for i := 0; i < max-len(train); i++ {
			for j := 0; j < 4; j++ {
				embedding := make([]float64, 256)
				input.Data = append(input.Data, embedding...)
			}
			output = append(output, 0)
		}
		output = append(output, 0)
		embedding, _ := PCA(input)
		embedding = Normalize(embedding)
		sa := SelfAttention(embedding, embedding, embedding)
		for i := 0; i < sa.Rows; i++ {
			in = append(in, sa.Data[i*sa.Cols:(i+1)*sa.Cols])
		}
	}

	forest := randomforest.Forest{}
	forest.Data = randomforest.ForestData{X: in, Class: output}
	fmt.Println("training...")
	forest.Train(1000)
	symbol := forest.Vote(in[5])
	fmt.Println(len(symbol))
	x, index := 0.0, 0
	for k, v := range symbol {
		if v > x {
			x, index = v, k
		}
	}
	fmt.Printf("\"%c\" \"%c\"\n", index, output[5])
}
