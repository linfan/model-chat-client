package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
)

// 定义请求体结构
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	Stream      bool      `json:"stream"`
}

// 定义配置结构
type ChatConfig struct {
	Url         string
	ApiKey      string
	Model       string
	Temperature float64
}

// 定义消息结构
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func getEnvWithDefault(key string, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		value = defaultValue
	}
	return value
}

// 从环境变量加载配置，增加可选参数 modelOverride
func loadConfig(model string) ChatConfig {
	temp := getEnvWithDefault("OPENAI_TEMPERATURE", "0.7")
	temperature, err := strconv.ParseFloat(temp, 64)
	if err != nil {
		log.Fatalf("无法解析温度参数: %v", err)
	}

	return ChatConfig{
		Url:         getEnvWithDefault("OPENAI_URL", "http://127.0.0.1:8000/v1/chat/completions"),
		ApiKey:      getEnvWithDefault("OPENAI_API_KEY", "none"),
		Model:       getEnvWithDefault("OPENAI_MODEL", model),
		Temperature: temperature,
	}
}

// 发送SSE请求并处理响应
func chat(messages []Message, conf ChatConfig) string {
	// 构造请求体
	reqBody := ChatRequest{
		Model:       conf.Model,
		Messages:    messages,
		Temperature: conf.Temperature,
		Stream:      true,
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		log.Fatalf("无法序列化请求体: %v", err)
	}

	// 创建HTTP请求
	req, err := http.NewRequest("POST", conf.Url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		log.Fatalf("无法创建HTTP请求: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", conf.ApiKey))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Fatalf("HTTP请求失败: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应状态码
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Fatalf("请求失败，状态码: %d, 响应: %s", resp.StatusCode, body)
	}

	isInThinking := false

	// 解析SSE流并增量打印内容
	reader := bufio.NewReader(resp.Body)
	answer := ""
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			log.Fatalf("读取SSE流失败: %v", err)
		}

		// 提取data字段
		if strings.HasPrefix(line, "data:") {
			data := strings.TrimSpace(line[5:])
			if data == "[DONE]" {
				break
			}

			// 解析JSON数据
			var event map[string]interface{}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				log.Printf("无法解析SSE事件: %v", err)
				continue
			}

			// 提取助手回复内容
			if choices, ok := event["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if delta, ok := choice["delta"].(map[string]interface{}); ok {
						if reasoningContent, _ := delta["reasoning_content"].(string); reasoningContent != "" {
							// 增量打印思考过程
							if !isInThinking {
								fmt.Printf("----------------- THINKING START -----------------\n")
								isInThinking = true
							}
							_, _ = os.Stdout.Write([]byte(reasoningContent))
						}
						if content, _ := delta["content"].(string); content != "" {
							// 增量打印内容
							if isInThinking {
								fmt.Printf("\n------------------ THINKING END ------------------\n\n")
								isInThinking = false
							}
							_, _ = os.Stdout.Write([]byte(content))
							answer += content
						}
					}
				}
			}
		}

		if err == io.EOF {
			break
		}
	}
	fmt.Println() // 打印换行符
	return answer
}

func main() {
	// 初始化对话历史
	var messages []Message

	// 解析命令行参数
	var filePath string
	flag.StringVar(&filePath, "f", "", "指定要读取的文件路径")
	var modelName string
	flag.StringVar(&modelName, "m", "qwen", "指定要使用的模型名称")
	flag.Parse()

	fmt.Println("* Using model:", modelName)
	chatConfig := loadConfig(modelName)

	// 如果有额外的参数，则将其作为用户提问并调用聊天接口
	if len(flag.Args()) > 0 {
		userInput := strings.Join(flag.Args(), " ")
		messages = append(messages, Message{Role: "user", Content: userInput})

		// 使用用户输入作为第一轮对话的输入
		firstAnswer := chat(messages, chatConfig)
		messages = append(messages, Message{Role: "assistant", Content: firstAnswer})
	}

	// 如果提供了文件路径，则读取文件内容作为初始提问并调用聊天接口
	if filePath != "" {
		fileContent, err := os.ReadFile(filePath)
		if err != nil {
			log.Fatalf("无法读取文件: %v", err)
		}
		messages = append(messages, Message{Role: "user", Content: string(fileContent)})

		// 使用文件内容作为第一轮对话的输入
		firstAnswer := chat(messages, chatConfig)
		messages = append(messages, Message{Role: "assistant", Content: firstAnswer})
	}

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("开始对话（输入'exit'退出）：")
	for {
		fmt.Print("> ")
		scanner.Scan()
		input := scanner.Text()

		if input == "exit" {
			break
		}

		// 添加用户消息到对话历史
		messages = append(messages, Message{Role: "user", Content: input})

		// 调用聊天接口时传递空字符串作为默认值
		answer := chat(messages, chatConfig)

		// 添加助手回复到对话历史（需要根据实际SSE返回内容更新）
		messages = append(messages, Message{Role: "assistant", Content: answer})
	}
}
