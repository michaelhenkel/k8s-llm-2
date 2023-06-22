use llm_chain::options;
use llm_chain::options::{ModelRef, Options};
use llm_chain::prompt::Conversation;
use llm_chain::output::StreamExt;
use llm_chain::{chains::conversation::Chain, executor, parameters, prompt, step::Step};
use std::{env::args, error::Error};
/// This example demonstrates how to use the llm-chain for few-shot prompting
///
/// This example can be seen as a "chain of thought"
///
/// Usage: cargo run --example few-shot
///
/// Make sure to have the env var 'LLAMA_MODEL_PATH' set
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = args().collect();
    let model_path = args[1].clone();

    let opts = options!(
        Model: ModelRef::from_path(model_path.clone()),
        ModelType: "llama",
        NThreads: 4_usize,
        
        MaxContextSize: 512_usize,
        MaxTokens: 0_usize,
        TopK: 40_i32,
        TopP: 0.95,
        TfsZ: 1.0,
        TypicalP: 1.0,
        Temperature: 0.8,
        RepeatPenalty: 1.1,
        RepeatPenaltyLastN: 64_usize,
        FrequencyPenalty: 0.0,
        PresencePenalty: 0.0,
        Mirostat: 0_i32,
        MirostatTau: 5.0,
        MirostatEta: 0.1,
        PenalizeNl: true,
        
        StopSequence: vec!["\n".to_string()]
    );

    /*
    let res = tokio::task::spawn(async {
        runner2(opts,"hello, what's your name?").await;
    });
    */
    let res = tokio::task::spawn(async {
        runner3(model_path).await;
    });
    let out = res.await;
    match out {
        Ok(_) => println!("OK"),
        Err(e) => println!("ERR {:?}", e),
    }
    Ok(())
}

async fn runner3(model_path: String) -> Result<(), Box<dyn std::error::Error>>{
    let opts = options!(
        //Model: ModelRef::from_path(model_path),
        //ModelType: "llama",
        NThreads: 4_usize,
        MaxContextSize: 512_usize,
        StopSequence: vec!["\n".to_string()]
    );

    let exec_1 = executor!(llama, opts.clone())?;

    let user_prompt =
        "Take the last letters of the words in '{{ full_name }}' and concatenate them";
    let res = Step::for_prompt_template(prompt!(user: user_prompt))
        .run(&parameters!().with("full_name", "Elon Musk"), &exec_1)
        .await?;
    println!("{} (zero-shot answer)", res.to_immediate().await?); // probably not correct
    let conversation = Conversation::new()
        .with_user_template(
            user_prompt,
            &parameters!().with("full_name", "Saquon Barkley"),
        )?
        .with_assistant("SB".to_string())
        .with_user_template(
            user_prompt,
            &parameters!().with("full_name", "Sean Connery"),
        )?
        .with_assistant("SC".to_string())
        .with_user_template(
            user_prompt,
            &parameters!().with("full_name", "Julius Ceasar"),
        )?
        .with_assistant("JC".to_string())
        .with_user_template(user_prompt, &parameters!().with("full_name", "Ding Liren"))?
        .with_assistant("DL".to_string());
    // build a chain, that has the above conversation stored in its state
    let mut chain = Chain::new_with_message_collection(&conversation);
    // Try the Elon Musk problem again
    // Define the step
    let step = Step::for_prompt_template(prompt!(user: user_prompt));
    // Execute the chain.
    let exec_2 = executor!(llama, opts)?;
    let res = chain
        .send_message(step, &parameters!().with("full_name", "Elon Musk"), &exec_2)
        .await?;
    println!("{} (few-shot CoT answer)", res.to_immediate().await?);
    Ok(())
}

async fn runner2(opts: Options, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("runner2");
    
    let exec = executor!(llama, opts.clone())?;

    let user_prompt =
        "show a code example of a struct in the Rust programming language";
    let res = Step::for_prompt_template(prompt!(user: user_prompt))
        .run(&parameters!(), &exec)
        .await?;
    let mut stream = res.as_stream().await?;
    while let Some(v) = stream.next().await {
        print!("{}", v);
    }
    println!(" ");
    Ok(())
}

async fn runner(opts: Options) -> Result<(), Box<dyn std::error::Error>> {
    println!("runner");
    /* 
    let exec_1_res = executor!(llama, opts.clone());
    let exec_1 = match exec_1_res {
        Ok(exec_1) => { exec_1 },
        Err(e) => { return Err(Box::new(e));},
    };
    
    
    let res = Step::for_prompt_template(prompt!(user: user_prompt))
        .run(&parameters!().with("full_name", "Elon Musk"), &exec_1)
        .await?;
    
    println!("{} (zero-shot answer)", res.to_immediate().await?); // probably not correct
    */

    let user_prompt =
    "Take the last letters of the words in '{{ full_name }}' and concatenate them";

    let conversation = Conversation::new()
        .with_user_template(
            user_prompt,
            &parameters!().with("full_name", "Saquon Barkley"),
        )?
        .with_assistant("SB".to_string());
    // build a chain, that has the above conversation stored in its state
    let mut chain = Chain::new_with_message_collection(&conversation);
    // Try the Elon Musk problem again
    // Define the step
    let step = Step::for_prompt_template(prompt!(user: user_prompt));
    // Execute the chain.
    let exec_2 = executor!(llama, opts)?;
    let res = chain
        .send_message(step, &parameters!().with("full_name", "Elon Musk"), &exec_2)
        .await?;
    println!("{} (few-shot CoT answer)", res.to_immediate().await?);
    Ok(())
}