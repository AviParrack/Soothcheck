# 1/6/25

- Someone got an LLM to [<u>create a pretty good stand-up routine for other LLMs</u>](https://x.com/AmandaAskell/status/1874873487355249151).

  - My first reaction was: wow, the LLM is pretty funny!

  - But I realize that that’s partially an illusion.

  - The LLM, working in tandem with the human, is pretty funny.

  - The human is deciding how to prompt the LLM, and which things the LLM produces that are compelling enough to share.

  - Presumably most of the output was pretty bland or uninspired, but we see only the highlights, from the human’s curatorial taste.

  - Humans only bother to share things they think are good. That curation is an act of creation.

  - The combination of a savvy, high taste human and an LLM are a powerful creative duo, much stronger than either would be alone.

  - Irreducibly co-creative acts.

  - The human as supervisor or editor to the LLM.

- The append-only conversation log to interact with LLMs leaves me wanting more.

  - In the past few months I’ve had a number of revelatory, multi-hour conversations with LLMs where I explore and pull on threads of my curiosity, in some cases leading to deep insights that reconfigure my mental model of the world or myself.

  - These conversations are long and meandering. They have various dead ends or odd paths that I later decide aren’t useful.

  - These aren’t like an essay or a simple conversation with a collaborator; they are creative, open-ended insight-generating interactions.

  - I’d love to be able to prune past parts of the conversation that I later realized were tangents, or fork a pre-existing conversation midway through and see how I could take it in new directions.

  - You can cobble these kinds of interactions together by starting new conversations, or using Anthropic’s projects, or manually cobbling together tools on top of the raw API.

    - It feels like I’m banging my head against a command line interface, wishing for a GUI.

  - The chat UI is too literal, like you’re talking to any other normal human.

    - In a human chat, it doesn’t make sense to edit previous messages or fork conversations.

    - But it’s only superficially a chat; it’s actually a way to drive an LLM to interesting outputs; a co-creative act that just so happens to manifest as though it’s a normal chat.

  - These kinds of open-ended exploratory conversations are just one killer use for LLMs, and even though they’re a pretty close fit for a conversation-style chat, it still feels constricting!

- Interacting with LLMs today is like doing computing before the GUI.

  - Clear raw potential that it takes specialist skills to unlock and apply.

  - The command line was a remnant of an older interaction type of teletype.

  - The GUI unlocked new kinds of interactions that only made sense in a world of interactive computers.

  - What will the equivalent of a GUI for an LLM be?

  - Creating the equivalent to the first GUI is a creative act wildly unlike building a new product in an existing paradigm. It requires multi-ply thinking.

- A nice little use for LLMs: discovering best practices.

  - For example, download three different popular templates for Project Management from Notion, and provide them to the LLM, and ask it to find commonalities.

  - The things that every good Project Management template agrees on is much more likely to be fundamentally, inescapably important in that domain, rather than just a random detail from that template’s creator.

  - The synthesis across lots of different examples is a time-consuming, boring task… but LLMs never get bored.

- LLMs are unlike normal programs in that they allow open-ended output.

  - Normal programs are close-ended: given similar inputs they will behave similarly.

  - The open-endedness of LLMs is what gives them their human-like creativity and ability to apply human-style common sense.

  - But the open-endedness is also what makes them hard to use in a process you want to be predictable.

  - Most of the applications of software today are in contexts where we want each run to be predictable, and as we evolve the software to handle more and more edge cases, to run reliably and increasingly without need for human intervention.

    - Software, when it’s finely tuned to its domain, gets increasingly tightly aligned with the domain and able to run by itself without intervention.

  - Creativity is a double edged sword; it might have a clever way to handle an unexpected input, or it might do something weird you don’t like.

  - Another option for using LLMs is not to have them live in the main loop of execution, but rather use them for continuous compilation.

  - Give a clear, convergent goal for the software to execute.

  - When something unexpected happens, call up the LLM and give it the error message and have it tweak the software to handle it.

  - The LLM is tuning close-ended software.

  - This kind of approach works well in cases where you could imagine writing an operator manual for a call center employee: close-ended, but requires a bit of human-level judgement.

- I love [<u>this tweet from Geoffrey Litt</u>](https://x.com/geoffreylitt/status/1875215219188011167):

  - "AI shines in these kinds of situations where “good enough is good enough”

  - Not the optimal workout plan, not the best workout app…

  - Just something reasonable, fully tailored to me, ready in minutes."

- Why do people pick the good enough option over the great one?

  - "good enough" has to be radically cheaper (in cost, friction etc) than a "great" option.

  - But I'd take a real good enough thing than a theoretical great thing (that I'd have to build, configure, etc) any day of the week.

  - Good enough is a tradeoff not with a theoretical best, but best *available.*

    - Best available for many use cases is "non existent" or "terrible".

    - A low bar.

  - Good enough is the toehold of being viable, worth using.

  - From there if there's an iterative process you can climb up to get to greatness.

- An insightful quote from Ilya Sutskever [<u>via an old Less Wrong post</u>](https://www.lesswrong.com/posts/nH4c3Q9t9F3nJ7y8W/gpts-are-predictors-not-imitators) about why LLMs are so powerful: “to learn to predict text, is to learn to predict the causal processes of which the text is a shadow,”

- Imagine a world where software is an afterthought.

  - Today software is so expensive to create and so that is where all the power accrues.

  - We send our data to aggregate on the software creator’s turf because the software is so expensive.

  - In a world where small bits of crappy software are effectively free, that dynamic will flip on its head.

- LLMs are the *input* to the next big thing.

  - LLMs are like electricity: a general technology that reconfigures the costs of key inputs, making wildly new kinds of things feasible.

  - LLMs are good for a number of things, like applying human-level common sense.

    - One application is that you can make crappy small bits of software effectively for free–a massive change.

  - Now that this input has gotten so cheap, it’s a long journey to figure out what to build with them.

    - So many things we take for granted as being infeasible are now feasible… but it will take time to discover which ones.

  - We are in the early innings!

  - The LLM model providers are the electricity providers.

    - Expensive, competitive, value-creating... but not necessarily a great business.

- My friend Anthea [<u>pointed out</u>](https://docs.google.com/document/d/1GrEFrdF_IzRVXbGH1lG0aQMlvsB71XihPPqQN-ONTuo/edit?disco=AAABaMlKotI) that my assertion that LLMs capture “all of society” is wrong.

  - LLMs give a slice of the content represented on the internet, which has a strong western, English bias.

  - She imagines the experience of someone from a different culture talking with an LLM to be like watching a Hollywood movie.

  - She predicts that as LLMs become more ingrained in society, the fact they reflect one particular culture so strongly will become a major issue.

- Specialists are more likely to go extinct.

  - A specialist is optimized for their niche.

  - That allows them to outcompete others in their niche, but be worse than others at competing outside their niche.

  - That means that specialists are less able to handle the disappearance of their niche.

  - A generalist (especially one that has a low caloric need, due to small size or metabolic rate) is more resilient; more likely to be viable no matter what happens in the environment.

- Tons of new species are created when a disruptive thing happens.

  - Disruptive in the world of biology might be a new land mass opens up.

  - In the world of technology it might mean a new disruptive technology, like electricity, jet engines, or LLMs.

    - The new disruptive technology reconfigures the “fitness landscape” of viable ideas, by radically changing a key dimension of cost.

  - With all of the new territory available, there is massive expansion as species diffuse out to find and colonize the newly available niches.

    - This is called “adaptive radiation”.

  - Later, once all of the niches are colonized, competition heats up within those niches.

  - Competition over scarce territory leads to more efficiency at the cost of variety.

  - This riff is inspired by this video: [<u>Why do Animals Look so Strange After Mass Extinctions</u>](https://youtu.be/1yTAKJFbPFU?si=5oGnMJXKiC0VsPby)

- A/B testing can get you stuck in fractal niches upon niches.

  - That is, weird local maxima within local maxima.

  - When you look up you are like “wait where are we? This obviously doesn’t make sense.”

  - A/B testing doesn’t have a vision, it’s just the path of least resistance.

  - If you follow the path of least resistance many many times, you end up in incoherent places, lost from any semblance of something bigger.

  - Just the "most".

  - The most what?

  - Whatever catchment basin you happened to start in.

- Our modern environment is dominated by supernormal stimuli.

  - Normally in evolutionary contexts, the stimuli an organism is exposed to in their environment changes at a similar speed as their own evolution.

    - For example, a butterfly being attracted to a flower that is co-evolving with it. As the flower ramps up the intensity of its stimuli, the butterfly coevolves to be less responsive to the stimuli.

    - Both evolutionary loops are in a biological substrate, which has a similar clock speed.

  - This allows their response to a stimuli to be well-calibrated to the stimuli.

  - But sometimes the environment can change much more quickly than the species can evolve, and produce “supernormal” stimuli that activate the response way more strongly than ideal.

  - When humans created language, which allowed mutating and passing down ideas orders of magnitude faster than biological evolution, it created a machine that could produce supernormal stimuli.

  - Modern commerce is extremely good at producing these supernormal stimuli that we find irresistible because they are orders of magnitude more potent than the stimuli we evolved for.

  - Examples include modern junk food and also our informational junk food.

  - Capitalism is great for giving users what they want, not what they *want* to want.

    - Why is there an obesity epidemic? Because capitalism figured out great ways to make irresistible food that our lizard brains can't say no to.

    - Capitalism is very good at giving us what we want: supernormal stimuli.

  - This riff is inspired by this video: [<u>Why are Lemurs Terrified of Predators that don't Exist?</u>](https://youtu.be/CXvR5v6MyQg?si=EJ96RcmH4dLd0tIU)

- A great new essay from my friend Ivan: [<u>Shallow Feedback Hollows You Out</u>](https://nothinghuman.substack.com/p/shallow-feedback-hollows-you-out).

  - It dives into why even novel, fresh thinkers over time become simplistic, hollowed out versions of themselves.

  - It feels to me like regression to the mean is an inescapable phenomena, perhaps as fundamental as (or directly implied by) the second law of thermodynamics.

- You shouldn't get mad at the weather.

  - Some phenomena are emergent, inescapable.

  - Nothing you do will change them.

  - All of the effort of being mad at the weather is wasted.

- Only non-believers bother with the details.

  - If you believe, you are willing to trust without looking closely, to give the benefit of the doubt.

  - Some creative endeavors simply require belief.

  - No amount of details can make someone believe.

  - Details and belief are two totally different kinds of things.

- The smarter you are, the easier you are to auto-nerd-snipe.

  - The world will be constantly nerd sniping you; you’re engaged and curious enough to see the interesting threads to pull on all around you.

- A product is about what people get. A community is about what people give.

  - A product and a community are two fundamentally different ends of a spectrum.

  - Scenius (the fullest embodiment of a community) is about putting the community above all else.

    - Everyone wants to be a part of, even if it takes personally expensive setbacks to participate.

- The job of a great manager is to bring out the best in their reports.

  - To not just tell them that they're great just the way they are, but to give them the space, encouragement, and drive to be not just good but great.

- Don’t be satisfied with a team dynamic that “makes it work”.

  - That can only give you middling results.

  - Make a team dynamic that makes it great.

  - That's the only way to achieve greatness together.

- Someone was telling me about how BMW thinks about their supply chain.

  - Instead of BMW killing themselves to make the best car, they have an ecosystem of suppliers killing themselves to be a competitive part of the best car.

  - This is a great example of harnessing the power of the swarm within yourself.

  - Although presumably it only works if you can figure out how to make the whole be significantly more than the sum of its parts.

  - For example, BMW’s brand implies that they vouch for any of the components used in the car; part of their creative act on the final car is making decisions about what components should exist and what is good enough.

- Nudging towards a better outcome only works on things that already have momentum.

  - To get momentum you have to have an asymmetric draw that pulls the group in a direction together.

- Over sufficient time horizons, you *will* get promoted to your level of incompetence.

  - The key question is: what do you do when that happens?

  - 1\) Find situational factors or an external villain to blame for your incompetence.

    - An approach that will accentuate and expand your incompetence by removing the ability to grow.

  - 2\) Step down from the role back to your level of competence.

    - An approach that just achieves stasis.

  - 3\) See it as a growth opportunity and challenge yourself to become competent.

    - Using the challenge to grow more competence, by taking the responsibility to grow on your own shoulders.

- You take the structures that you've always been embedded in for granted.

  - Just the way it works, the water you swim in.

  - Don’t even realize it could work a different way somewhere else.

- Constraints breed coherence because everything in the same neighborhood has the same constraints.

  - So if they respond to them in a similar way they will have coherent behavior without any explicit coordination.

  - Coordinated, coherent behavior, automatically.

- An expert in their element will look to novices like they're winging it.

  - Because they're playful and flexible.

  - But that's wrong, they are surfing it, totally in control, responding to the situation as it evolves, in a pitch perfect, calibrated way.

<!-- -->

- In a disruptive environment, throw out the mature playbook and shoot for the moon.

  - The rules have changed in ways that won’t become obvious for some time.

  - Following the rules will get you stuck in local maxima or even worse, if the rules no longer work the way you thought they did in the old world.

  - After disruptive events, reality reconfigures, and massive new outcomes become possible.

  - The smartest thing is to not be too beholden to playbooks and rules from the pre disruptive context.

- It takes multiple shots to invent the future.

  - You can’t do it in a single shot.

  - If you have a spotlight on you, it’s hard to get multiple shots.

- If you're in the woods surviving on your own, you learn feral techniques that make it so you can never leave the woods.

  - If you just listen to the logic of the woods you'll learn to survive, but also become feral and lose your soul.

- I find video conferences an order of magnitude more productive than an audio-only phone call.

  - One reason is the audio quality.

  - But an even bigger part is that with video you can get real time feedback of how the other person is receiving what you’re saying.

  - You can verify that they are listening and absorbing, and know if you need to retransmit, slow down, or speed up.

  - You get a live signal that tells you how fast to unspool your thoughts so they get it, what you need to modulate to make it land as you intended.

  - On the phone you can't tell if they're distracted, or the connection dropped out.

  - You have to wait until you’re done with your utterance to get any signal about how it was received.

- The fusion of two people prevents people from coming together because they're stuck together.

  - You need to be individuals, collaborating, dancing together to have a dynamic outcome.

  - Being individual while still being a pair together.

  - Not a boring, fused entity of double the size, but a different kind of thing than any individual could be.

- Process only makes sense once you’ve got to a normal operating cadence.

  - This can only happen after a “PMF” moment.

  - Before that, it’s all chaos all the time, constant black swans.

  - The amount of process should be inversely proportional to the amount of surprise, and before that moment of traction, surprise is constant.

- Some environments are large enough that good fences can make good neighbors.

  - But in a smaller, more chaotic environment, whether you like it or not, you're cohabitating.

  - There are no clear lines.

  - That's much more similar to family dynamics than neighbor dynamics.

- Nobody can have *full* ownership of anything ever.

  - Everyone has a boss. Even if you don’t, the market, the constraints, your investors all might be your ‘boss’.

  - None of us are an island.

  - The constraints define what we can do... but also give us a structure to operate within.

  - "I'd be able to do this if freed from constraints" is an impossible bar to clear.

  - You'll chase it, fruitlessly.

  - Constraints give us structure to climb on, otherwise it would be a free floating dead space with no way to get leverage and make any movement.

- You can’t force someone to be engaged.

  - To be engaged is to go above and beyond, act like an owner, invest discretionary effort.

  - Going above and beyond is a choice that has to come from within.

  - You can create the conditions where people are more likely to make the choice but you’re not able to ever compel it.

- Blame gets in the way of acknowledging response-ability, which is what is necessary to grow.

  - The ideal response is: “I didn’t choose the constraints I was within, but I did choose what to do within those constraints, and I accept responsibility for those choices.”

  - When you accept responsibility for those choices you can grow.

  - Now you can ask yourself how you might have chosen differently within those constraints that would have led to better outcomes.

- A vampirical way to make money: take something with a life force and convert that force to profit.

  - Converting the life force to profit is a way to make a quick buck, but it leads the living thing to become a husk of themselves, inches from death.

  - In the meantime, you might congratulate yourself for how much value you’re creating.

  - No, you’re *harvesting* it, in the most vampiric way you can.

  - People who see the business not as an end to create something of value in society, but only as a means to make money.

- Rigor does not mean more detail.

  - Rigor means that the closer someone looks at the answer, the more compelling it becomes.

  - Sometimes very simple conclusions can be enormously clarifying: inescapable, obvious, once you see it you can’t unsee it.

  - In some cases, the simple words of “not yet” are extremely clarifying.

    - That you don’t need to know the answer to some fuzzy, hard to distill thing saves you tons of time.

  - You know you’re getting closer to rigor and clarity as things get simpler, not more complex.

- Disconfirming evidence is simply a surprise.

  - A thing not predicted by your model, whose existence implies your model, as it exists, is wrong.

  - It is only by realizing your model is not right that you can hope to improve it.

- A high functioning environment will constantly be unearthing and collaboratively processing disconfirming evidence.

  - A dysfunctional environment will be constantly ignoring or sweeping under the rug disconfirming evidence.

    - “Look, we clearly don’t disagree, because we’re both being nice to each other when we talk!”

  - It is only with sources of disconfirming evidence that a thing can become great.

- Abrasion is a process to help shave off little slivers of disconfirming evidence.

  - It is a process to produce these little bits of disconfirming evidence.

  - Tension is not abrasion.

  - Tension is stressful but does not necessarily turn up the raw material to become great.

  - Abrasion brings the disconfirming evidence to the surface.

  - Healthy abrasion helps make sure the group is actually dealing with disconfirming evidence.

- Three ingredients for greatness: abrasion, agility, and resolution.

  - Abrasion to turn up little slivers of disconfirming evidence.

  - Agility to not get bogged down and over-complexify, to be willing to say “not yet” or focus on pragmatic responses.

  - Resolution to absorb and distill the disconfirming evidence into a new shared, improved model.

  - These three ingredients allow high quality, confident momentum in a team.

- Some people are naturally abrasive.

  - In the best environments that abrasion is the source of disconfirming evidence that makes the team great.

    - People on the team find that abrasion to be empowering.

  - In the worst environments that abrasion is pure intimidation.

    - It makes people feel overwhelmed, and they just defensively crouch.

  - The question is how to create an environment where the naturally abrasive person can cause the former, not the latter.

- That “aha” feeling is apparently called [<u>bisociation</u>](https://www.themarginalian.org/index.php/2013/05/20/arthur-koestler-creativity-bisociation/).

  - It’s the feeling of a wave function of possible interpretations all collapsing into one clear answer with a jolt.

  - In some cases it’s delightful and engaging, it’s naturally viral.

  - In other cases, it’s unsettling and odd, like the AI videos with weird interpretations of the scene, or the dolly-zoom from Jaws.

    - The feeling of a figure-ground inversion; something that flipped that you didn’t even realize was a thing, let alone a thing that could be flipped.

  - The aha moment is adjacent to humor.

    - They are both the creative work, the delightful swoosh of intentional figure ground inversion.

- Apparently some people have a [<u>need for cognition</u>](https://en.wikipedia.org/wiki/Need_for_cognition).

  - Most people find cognition or critical thinking to be [<u>exhausting, boring, a chore</u>](https://thedebrief.org/new-study-confirms-critical-thinking-is-mentally-draining-and-inherently-unpleasant/).

  - Some weirdos (like me) love the feeling of cognition so much that they have a need for it.

    - I chase the “aha” moment, I’m addicted!

  - I also realize that a lot of the people I seek out as intellectual collaborators also have a deep need for cognition.

  - People with this need make wonderful open-ended “yes, and” collaborative debate partners, because they get more energy the more you explore the space.

- I like [<u>Jake Dahn’s comment</u>](https://docs.google.com/document/d/1GrEFrdF_IzRVXbGH1lG0aQMlvsB71XihPPqQN-ONTuo/edit?disco=AAABaMlKohc) on last week’s riff about play so much I’m going to quote it in full:

  - “A buddhist flavored lens on why play is where you do your best work:

  - The phenomenon of doing your best work through play is tied up in 2 buddhist concepts.

  - Vedanā (feeling tone), and Shoshin (beginners mind).

  - When you're in a professional setting, you're often chasing deadlines or trying to meet some external goal. This alters the vedanā, and scopes your attention into a mode of "seriousness".

  - Similarly, when you inevitably become the resident expert \[insert topic here\], your collaborators rely on you to be "the person that knows the thing", you can't practice shoshin, you are forced to be the stable channel of wisdom or best practice.

  - I suspect the only way to consistently practice play during the work day is to schedule unstructured play time.

  - Make space, not plans.”