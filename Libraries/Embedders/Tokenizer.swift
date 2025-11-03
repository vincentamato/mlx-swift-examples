// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers
import SentencepieceTokenizer

/// Wrapper that adapts SentencepieceTokenizer to the Tokenizer protocol
public class SentencePieceTokenizerWrapper: Tokenizer {
    private let tokenizer: SentencepieceTokenizer

    public init(modelPath: String) throws {
        self.tokenizer = try SentencepieceTokenizer(modelPath: modelPath)
    }

    public func encode(text: String) -> [Int] {
        return (try? tokenizer.encode(text)) ?? []
    }

    public func decode(tokens: [Int]) -> String {
        return (try? tokenizer.decode(tokens)) ?? ""
    }

    public func callAsFunction(_ text: String) -> [Int] {
        return encode(text: text)
    }
}

public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
    // Get the model directory
    let modelDirectory: URL
    switch configuration.id {
    case .id(_):
        modelDirectory = configuration.modelDirectory(hub: hub)
    case .directory(let directory):
        modelDirectory = directory
    }

    // Check if tokenizer.model exists (SentencePiece tokenizer)
    let sentencePieceTokenizerPath = modelDirectory.appendingPathComponent("tokenizer.model")
    if FileManager.default.fileExists(atPath: sentencePieceTokenizerPath.path) {
        // Use SentencepieceTokenizer wrapper for models like EmbeddingGemma
        return try SentencePieceTokenizerWrapper(modelPath: sentencePieceTokenizerPath.path)
    } else {
        // Fall back to PreTrainedTokenizer for other models
        let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
            configuration: configuration, hub: hub)

        return try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}

func loadTokenizerConfig(configuration: ModelConfiguration, hub: HubApi) async throws -> (
    Config, Config
) {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config: LanguageModelConfigurationFromHub

    switch configuration.id {
    case .id(let id):
        do {
            // the load can fail (async when we try to use it)
            let loaded = LanguageModelConfigurationFromHub(
                modelName: configuration.tokenizerId ?? id, hubApi: hub)
            _ = try await loaded.tokenizerConfig
            config = loaded
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain
                && nserror.code == NSURLErrorNotConnectedToInternet
            {
                // Internet connection appears to be offline -- fall back to loading from
                // the local directory
                config = LanguageModelConfigurationFromHub(
                    modelFolder: configuration.modelDirectory(hub: hub), hubApi: hub)
            } else {
                throw error
            }
        }
    case .directory(let directory):
        config = LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
    }

    guard let tokenizerConfig = try await config.tokenizerConfig else {
        throw EmbedderError(message: "missing config")
    }
    let tokenizerData = try await config.tokenizerData
    return (tokenizerConfig, tokenizerData)
}
