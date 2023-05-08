package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
)

type openAIChatRequest struct {
	Model    string              `json:"model"`
	Messages []openAIChatMessage `json:"messages"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAICompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   int      `json:"max_tokens"`
	Temperature float64  `json:"temperature"`
	TopP        float64  `json:"top_p"`
	N           int      `json:"n"`
	Stream      bool     `json:"stream"`
	Logprobs    *int     `json:"logprobs,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

type generation struct {
	WorkerID   string `json:"worker_id"`
	WorkerName string `json:"worker_name"`
	Model      string `json:"model"`
	State      string `json:"state"`
	Text       string `json:"text"`
	Seed       int    `json:"seed"`
}

type openAIChatResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int                `json:"created"`
	Choices []openAIChatChoice `json:"choices"`
	Usage   openAIUsage        `json:"usage"`
}

type openAIChatChoice struct {
	Index        int               `json:"index"`
	Message      openAIChatMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAICompletionResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int                      `json:"created"`
	Model   string                   `json:"model"`
	Choices []openAICompletionChoice `json:"choices"`
	Usage   openAIUsage              `json:"usage"`
}

type openAICompletionChoice struct {
	Text         string      `json:"text"`
	Index        int         `json:"index"`
	Logprobs     interface{} `json:"logprobs,omitempty"`
	FinishReason string      `json:"finish_reason"`
}

type koboldAIRequest struct {
	Prompt         string   `json:"prompt"`
	Models         []string `json:"models"`
	TrustedWorkers bool     `json:"trusted_workers"`
	Params         params   `json:"params"`
}

type params struct {
	MaxContextLength int `json:"max_context_length"`
	MaxLength        int `json:"max_length"`
}

type koboldAIPollResponse struct {
	Finished      int          `json:"finished"`
	Processing    int          `json:"processing"`
	Restarted     int          `json:"restarted"`
	Waiting       int          `json:"waiting"`
	Done          bool         `json:"done"`
	Faulted       bool         `json:"faulted"`
	WaitTime      int          `json:"wait_time"`
	QueuePosition int          `json:"queue_position"`
	Kudos         float32      `json:"kudos"`
	IsPossible    bool         `json:"is_possible"`
	Generations   []generation `json:"generations"`
}

type koboldAIAsyncResponse struct {
	ID      string `json:"id"`
	Message string `json:"message"`
}

const (
	koboldAPIURL    = "https://horde.koboldai.net/api/v2/generate/text/async"
	koboldStatusURL = "https://horde.koboldai.net/api/v2/generate/text/status/"
	defaultAPIKey   = "0000000000"
)

func main() {
	router := mux.NewRouter()

	router.HandleFunc("/v1/chat/completions", chatCompletionHandler).Methods("POST")
	router.HandleFunc("/v1/completions", completionHandler).Methods("POST")

	http.ListenAndServe(":8080", router)
}

func chatCompletionHandler(w http.ResponseWriter, r *http.Request) {
	apiKey := r.Header.Get("Authorization")
	if apiKey == "" {
		apiKey = defaultAPIKey
	} else {
		apiKey = strings.TrimPrefix(apiKey, "Bearer ")
	}

	var chatReq openAIChatRequest
	if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
		fmt.Fprintf(os.Stderr, "Error decoding request body: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	koboldReq := convertOpenAIChatRequestToKobold(chatReq)
	koboldResp, err := callKoboldAPI(koboldReq, apiKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error calling Kobold API: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	chatResp := convertKoboldResponseToOpenAIChatResponse(koboldResp)

	fmt.Fprintln(os.Stdout, chatResp)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(chatResp)
}

func completionHandler(w http.ResponseWriter, r *http.Request) {
	apiKey := r.Header.Get("Authorization")
	if apiKey == "" {
		apiKey = defaultAPIKey
	} else {
		apiKey = strings.TrimPrefix(apiKey, "Bearer ")
	}

	var completionReq openAICompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&completionReq); err != nil {
		fmt.Fprintln(os.Stderr, err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	koboldReq := convertOpenAICompletionRequestToKobold(completionReq)
	koboldResp, err := callKoboldAPI(koboldReq, apiKey)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	completionResp := convertKoboldResponseToOpenAICompletionResponse(koboldResp)
	//w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(completionResp)
}

func convertOpenAIChatRequestToKobold(chatReq openAIChatRequest) koboldAIRequest {
	prompt := ""
	for _, message := range chatReq.Messages {
		prompt += message.Role + ": " + message.Content + "\n"
	}

	return koboldAIRequest{
		Prompt:         prompt,
		Models:         []string{chatReq.Model},
		TrustedWorkers: false,
		Params: params{
			MaxContextLength: 1024,
			MaxLength:        100,
		},
	}
}

func convertOpenAICompletionRequestToKobold(completionReq openAICompletionRequest) koboldAIRequest {
	return koboldAIRequest{
		Prompt:         completionReq.Prompt,
		Models:         []string{completionReq.Model},
		TrustedWorkers: false,
		Params: params{
			MaxContextLength: 1024,
			MaxLength:        completionReq.MaxTokens,
		},
	}
}

func callKoboldAPI(koboldReq koboldAIRequest, apiKey string) (koboldAIPollResponse, error) {
	fmt.Printf("Req: %+v\n", koboldReq)
	reqBody, err := json.Marshal(koboldReq)
	if err != nil {
		_, file, line, _ := runtime.Caller(0)
		errMsg := fmt.Sprintf("Error while marshal request at %s:%d: %v", file, line, err)
		fmt.Fprintln(os.Stderr, errMsg)
		return koboldAIPollResponse{}, err
	}

	req, err := http.NewRequest("POST", koboldAPIURL, strings.NewReader(string(reqBody)))
	if err != nil {
		_, file, line, _ := runtime.Caller(0)
		errMsg := fmt.Sprintf("Error while creating request at %s:%d: %v", file, line, err)
		fmt.Fprintln(os.Stderr, errMsg)
		return koboldAIPollResponse{}, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("apikey", apiKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	fmt.Fprintln(os.Stdout, "Sending request to horde")
	if err != nil {
		_, file, line, _ := runtime.Caller(0)
		errMsg := fmt.Sprintf("Error while querying horde at %s:%d: %v", file, line, err)
		fmt.Fprintln(os.Stderr, errMsg)
		return koboldAIPollResponse{}, err
	}
	defer resp.Body.Close()

	var jsonResponse koboldAIAsyncResponse
	err = json.NewDecoder(resp.Body).Decode(&jsonResponse)
	if err != nil {
		_, file, line, _ := runtime.Caller(0)
		errMsg := fmt.Sprintf("Error occurred at %s:%d: %v", file, line, err)
		fmt.Fprintln(os.Stderr, errMsg)
		return koboldAIPollResponse{}, err
	}
	fmt.Printf("Resp: %+v\n", jsonResponse)

	fmt.Fprintln(os.Stdout, "Polling horde with job id ", jsonResponse.ID)
	result, err := pollKoboldAPI(jsonResponse.ID)
	if err != nil {
		errMsg := fmt.Sprintf("Error polling horde: %v", err)
		fmt.Fprintln(os.Stderr, errMsg)
		return koboldAIPollResponse{}, err
	}

	return result, nil
}

func pollKoboldAPI(id string) (koboldAIPollResponse, error) {
	statusEndpoint := koboldStatusURL + id

	for {
		time.Sleep(2 * time.Second)

		resp, err := http.Get(statusEndpoint)
		if err != nil {
			errMsg := fmt.Sprintf("Error polling horde GET: %v", err)
			fmt.Fprintln(os.Stderr, errMsg)
			return koboldAIPollResponse{}, err
		}
		defer resp.Body.Close()

		var jsonResponse koboldAIPollResponse
		fmt.Fprintln(os.Stdout, jsonResponse)
		err = json.NewDecoder(resp.Body).Decode(&jsonResponse)
		if err != nil {
			errMsg := fmt.Sprintf("Error polling horde decode: %v", err)
			fmt.Fprintln(os.Stderr, errMsg)
			return koboldAIPollResponse{}, err
		}
		fmt.Printf("Resp: %+v\n", jsonResponse)

		if jsonResponse.Done {
			return jsonResponse, nil
		}
	}
}

func convertKoboldResponseToOpenAIChatResponse(koboldResp koboldAIPollResponse) openAIChatResponse {
	responseText := koboldResp.Generations[0].Text
	assistantMessage := openAIChatMessage{
		Role:    "assistant",
		Content: responseText,
	}

	id, _ := uuid.NewUUID()
	return openAIChatResponse{
		ID:      id.String(),
		Object:  "chat.completion",
		Created: int(time.Now().Unix()),
		Choices: []openAIChatChoice{
			{
				Index:        0,
				Message:      assistantMessage,
				FinishReason: "stop",
			},
		},
		Usage: openAIUsage{
			PromptTokens:     len(koboldResp.Generations[0].Text),
			CompletionTokens: len(responseText),
			TotalTokens:      len(koboldResp.Generations[0].Text) + len(responseText),
		},
	}
}

func convertKoboldResponseToOpenAICompletionResponse(koboldResp koboldAIPollResponse) openAICompletionResponse {
	responseText := koboldResp.Generations[0].Text

	id, _ := uuid.NewUUID()
	return openAICompletionResponse{
		ID:      id.String(),
		Object:  "text.completion",
		Created: int(time.Now().Unix()),
		Choices: []openAICompletionChoice{
			{
				Text:         responseText,
				Index:        0,
				Logprobs:     nil,
				FinishReason: "stop",
			},
		},
		Model: "davinci-codex",
		Usage: openAIUsage{
			PromptTokens:     0,
			CompletionTokens: 0,
			TotalTokens:      0,
		},
	}
}
